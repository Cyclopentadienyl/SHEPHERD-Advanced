"""
The model spec: what a model bundle is, written down so it can be rebuilt.
=========================================================================
One versioned, self-describing record per checkpoint
(`docs/working/scorer-retraining/README.md` §3–§4). The one builder
(`src.models.bundle.builder.build_bundle`) reads nothing else: no weight name,
no state-dict shape and no training flag decides what is built.

**Plain data, strictly read.** `ModelSpec.to_dict()` gives strings, numbers,
booleans, `None` and nested dicts/lists only, so a checkpoint carrying it still
loads with `weights_only=True`. `ModelSpec.from_dict()` refuses a record of
another version, an unknown key at any level and a missing required section:
a spec it cannot read completely is never half-read. **Every field holds the
type its dataclass declares** (`_require_declared_types`): a flag is `true` or
`false`, an integer is not a boolean or a float, a number is a finite number,
never a string or a boolean, a text is a string; nothing is cast into its type.

**Provisional defaults carry their source** (README §0.3). `REFERENCE_SOURCES`
names where each value of the reference recipe comes from; a spec built from
`reference_spec()` keeps that map in `sources`, and a value changed by an
operator is recorded as such by whoever changes it.

Module: src/models/bundle/spec.py
"""
from __future__ import annotations

import math
import reprlib
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import lru_cache
from typing import (Any, Dict, Mapping, Optional, Sequence, Tuple, Union, get_args, get_origin,
                    get_type_hints)

#: Bumped when a field's meaning changes or a field is removed. A reader never
#: guesses a field an older or newer version may have meant.
SPEC_VERSION = 1

from src.config.model_types import ENCODER_KINDS  # noqa: E402  (torch-free)

LEARNING_STAGES = ("base", "finetune")
TASKS = ("disease", "gene", "patients_like_me")

#: Where each value of the reference recipe comes from (README §2). A source is
#: never a claim that the value is calibrated on this project's data.
UPSTREAM = "mims-harvard/SHEPHERD@e95433a"
REFERENCE_SOURCES: Dict[str, str] = {
    "input.dim": f"{UPSTREAM} run_pretrain.sh:6-17 (nfeat 2048)",
    "input.init": f"{UPSTREAM} node_embedder_model.py:62 (nn.Embedding default N(0,1))",
    "encoder.layers": f"{UPSTREAM} node_embedder_model.py:77 (n_layers 3)",
    "encoder.hidden": f"{UPSTREAM} run_pretrain.sh (hidden 256; first layer doubled, node_embedder_model.py:58)",
    "encoder.output": f"{UPSTREAM} run_pretrain.sh (output 128)",
    "encoder.heads": f"{UPSTREAM} run_pretrain.sh (n_heads 4)",
    "encoder.dropout": f"{UPSTREAM} run_pretrain.sh (dropout 0.3)",
    "encoder.concat": f"{UPSTREAM} node_embedder_model.py:81-85 (GATv2Conv concat default; README C1)",
    "encoder.attention": "PyG 1.7.2 GATv2Conv defaults: dropout 0, add_self_loops, no shared weights, slope 0.2 (README U4)",
    "encoder.norm": f"{UPSTREAM} hparams.py:25 batch_layer; node_embedder_model.py:135-143 (README C2, §2.8)",
    "encoder.activation_slope": f"{UPSTREAM} node_embedder_model.py:140 (F.leaky_relu default 0.01)",
    "lp_head": f"{UPSTREAM} decoders.py bilinear + tanh; node_embedder_model.py:416-425 hinge, margin 1",
    "lp_head.relation_init": f"{UPSTREAM} node_embedder_model.py:114 (xavier_uniform, leaky_relu gain)",
    "patient_encoder": f"{UPSTREAM} train.py:67-68 (3 layers, 8 heads); torch TransformerEncoderLayer defaults",
    "patient_encoder.pooling": "allennlp 2.4.0 BilinearAttention, bias dropped (README U11)",
    "task_head.projection_slope": f"{UPSTREAM} hparams.py:96 (leaky_relu 0.1)",
    "task_head.distance": "pytorch-metric-learning 0.9.98 LpDistance(power=2), normalised (README U10)",
    "task_head.gene": f"{UPSTREAM} gp_aligner.py:95-101 (EMBSIM); add_spl_to_patients.py:104-109 (bounds); "
                      "eta 0.5 from run_causal_gene_discovery.sh",
    "task_head.augmentation": "paper Eq 10, K 3 and omega 0.7 as provisional defaults (README F10)",
    "graph_view": "paper Methods (dedup, self-loops removed, largest connected component); README G1",
}


class SpecError(ValueError):
    """A spec that cannot be read or built as written."""


def _section(section: str, data: Any) -> Mapping[str, Any]:
    """`data`, read as the spec's section `section`: a mapping whose keys are
    names, or a `SpecError` naming the section. Checked before anything is
    asked of it, so a value with no one answer to `in`, to sorting or to
    `dict()` (a tensor, a list, a key of another kind) is refused, never asked."""
    if not isinstance(data, Mapping):
        raise SpecError(f"model spec section {section!r} is {type(data).__name__}, not a mapping")
    for key in data:
        if not isinstance(key, str):
            raise SpecError(f"model spec section {section!r} has a key that is not a name "
                            f"({type(key).__name__})")
    return data


def _check_keys(section: str, data: Mapping[str, Any], allowed: Tuple[str, ...],
                required: Tuple[str, ...] = ()) -> None:
    _section(section, data)
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise SpecError(f"model spec section {section!r} has unknown key(s) {unknown}; "
                        "a spec is read completely or refused")
    missing = sorted(set(required) - set(data))
    if missing:
        raise SpecError(f"model spec section {section!r} lacks {missing}")


def _from(cls, section: str, data: Mapping[str, Any]):
    names = tuple(f.name for f in fields(cls))
    _check_keys(section, data, names, names)
    return cls(**{name: data[name] for name in names})


#: How much of a refused value a reason shows.
_SHOWN_LIMIT = 60


class _Brief(reprlib.Repr):
    """`reprlib.Repr`, reading a mapping, list, tuple or set of a subclass (an
    `OrderedDict`, a `torch.Size`) as one, so no container is shown in full."""

    def repr1(self, x: Any, level: int) -> str:
        for kind in (dict, list, tuple, set, frozenset):
            if isinstance(x, kind):
                return getattr(self, f"repr_{kind.__name__}")(x, level)
        return super().repr1(x, level)


#: A container as a reason shows it: a few levels and a few values of each.
_BRIEF = _Brief(maxlevel=3, maxdict=6, maxlist=6, maxtuple=6, maxset=6, maxfrozenset=6,
                maxstring=_SHOWN_LIMIT, maxlong=_SHOWN_LIMIT, maxother=_SHOWN_LIMIT)


def _shown(value: Any) -> str:
    """`value` as a reason shows it, whatever it is. A container is shown a
    few levels and values deep (`_BRIEF`), since `repr` walks all of one: a
    list nested past the interpreter's depth, which the safe loader reads,
    raised `RecursionError`, and one held many times over at each level would
    be written out for ever. A value whose text cannot be made (an integer
    longer than Python turns into text) is shown by its kind."""
    try:
        text = (_BRIEF.repr(value) if isinstance(value, (dict, list, tuple, set, frozenset))
                else repr(value))
    except (RecursionError, ValueError):
        return f"a {type(value).__name__} too large to show"
    return text if len(text) <= _SHOWN_LIMIT else text[:_SHOWN_LIMIT - 3] + "..."


@lru_cache(maxsize=None)
def declared_fields(cls: type) -> Tuple[Tuple[str, Any], ...]:
    """`cls`'s fields with their declared types, resolved: what the one type
    rule reads (`_require_declared_types`), and what the Training page's base
    form generates a section's fields from (`training_console.spec_form_fields`),
    so a field a section declares is a field the form offers."""
    hints = get_type_hints(cls)
    return tuple((f.name, hints[f.name]) for f in fields(cls))


def _require_declared_types(section: Any, prefix: str = "") -> None:
    """**Every field of a spec holds the type its dataclass declares**, in
    every section and every nested one: the one type rule, run first by
    `ModelSpec.validate`, the gate every spec passes (read from a record or
    a configuration by `from_dict`, or built in code). The section rules that
    follow check ranges and allowed values only.

    A `bool` field takes `true` or `false`; an `int` field an integer that is
    not a boolean; a `float` field a finite number (an integer or a float)
    that is not a boolean; a `str` field a string; a `Tuple` field a tuple (a
    record's list, which `from_dict` makes one) whose elements each pass
    their declared type, and as many as declared when the length is fixed; a
    `Dict` field a mapping whose keys and values pass theirs; a section field
    that section; `Optional[X]` also takes None. Nothing is cast: the string
    "false" is truthy, "0.1" is not a number and `true` is the integer 1, so
    read as one any of them would build a model its record does not describe
    (the reviewer's K8-1 recheck: `encoder.add_self_loops: "false"` trained
    with self-loops, `encoder.attention_slope: "0.2"` failed in the run's
    first step, `patient_encoder.transformer_layers: true` built one layer)."""
    for name, declared in declared_fields(type(section)):
        _require_type(f"{prefix}{name}", getattr(section, name), declared)


def _require_type(name: str, value: Any, declared: Any) -> None:
    """Refuse `value` of the field `name` unless it holds `declared`."""
    origin, args = get_origin(declared), get_args(declared)
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return
        members = [arg for arg in args if arg is not type(None)]
        if len(members) != 1:
            raise SpecError(f"model spec field {name!r} is declared as {declared!r}; the type "
                            "rule reads one type, optionally None")
        declared = members[0]
        origin, args = get_origin(declared), get_args(declared)
    if declared is bool:
        if not isinstance(value, bool):
            raise SpecError(f"{name} must be a boolean (true or false), not {_shown(value)}")
    elif declared is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpecError(f"{name} must be an integer, not {_shown(value)}")
    elif declared is float:
        if not _finite_number(value):
            raise SpecError(f"{name} must be a finite number, not {_shown(value)}")
    elif declared is str:
        if not isinstance(value, str):
            raise SpecError(f"{name} must be a string, not {_shown(value)}")
    elif origin is tuple:
        if not isinstance(value, tuple):
            raise SpecError(f"{name} must be a list (a tuple in a spec built in code), "
                            f"not {_shown(value)}")
        if len(args) == 2 and args[1] is Ellipsis:
            args = (args[0],) * len(value)
        elif len(value) != len(args):
            raise SpecError(f"{name} must hold {len(args)} values, not {len(value)}")
        for index, (element, kind) in enumerate(zip(value, args)):
            _require_type(f"{name}[{index}]", element, kind)
    elif origin is dict:
        if not isinstance(value, dict):
            raise SpecError(f"{name} must be a mapping, not {_shown(value)}")
        key_kind, value_kind = args
        for key, element in value.items():
            _require_type(f"{name} key {_shown(key)}", key, key_kind)
            _require_type(f"{name}[{_shown(key)}]", element, value_kind)
    elif is_dataclass(declared):
        if not isinstance(value, declared):
            raise SpecError(f"{name} must be a {declared.__name__} section, not {_shown(value)}")
        _require_declared_types(value, f"{name}.")
    else:
        raise SpecError(f"{name} declares {declared!r}, which the spec's type rule does not "
                        "know")


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        return False


def _record_tuple(value: Any) -> Any:
    """A record's list as the tuple its field declares; anything else as it
    is, for the type rule to refuse (a string or a mapping is no list)."""
    return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class GraphViewSpec:
    """The reference graph view's rule (README G1, G5).

    `exclude_relations` names relation keys (`"src|rel|dst"`) left out before
    the view is formed; a relation's reverse leaves with it.
    """

    rule: str = "reference/v1"
    drop_native_self_loops: bool = True
    deduplicate: bool = True
    largest_component: bool = True
    exclude_relations: Tuple[str, ...] = ()

    def validate(self) -> None:
        if self.rule != "reference/v1":
            raise SpecError(f"unknown graph view rule {self.rule!r}")
        if any(r.count("|") != 2 for r in self.exclude_relations):
            raise SpecError("graph_view.exclude_relations holds 'src|rel|dst' keys")


@dataclass(frozen=True)
class InputSpec:
    """The input module: a learned node table, optionally plus a node-type
    embedding (README U2, §3.3 S2)."""

    kind: str = "learned_table"
    dim: int = 2048
    init: str = "normal"
    type_embedding: bool = False

    def validate(self) -> None:
        if self.kind != "learned_table":
            raise SpecError(f"unknown input kind {self.kind!r}")
        if self.init != "normal":
            raise SpecError(f"unknown input init {self.init!r}")
        _positive_int("input.dim", self.dim)


@dataclass(frozen=True)
class EncoderSpec:
    """The GraphEncoder (README C1, C2, U1, U3, U4, §2.8).

    The fields are read by every encoder kind; **what a kind makes of them is
    its own declaration's** (`EncoderDeclaration`, `encoder_declaration`): the
    rules a spec of that kind obeys beyond the shared field checks below, its
    output width (`width`, which the LP head and the task heads read), the
    computations it offers and the sampler hops it needs (`hops_problem`). No
    shared check reads one kind's rule.

    For `gatv2`, the one kind this build declares and builds, the layer widths
    follow the source: the first layer's hidden size is
    `hidden * first_hidden_multiplier`, the middle ones `hidden`, the last
    `output`; every layer has `heads` heads, concatenated, so the encoder's
    output width is `output * heads`, and the sampler takes one hop per layer.
    A kind `ENCODER_KINDS` names without a declaration here is read, and
    refused when a run of it starts (`hops_problem`) and when it is built.
    """

    kind: str = "gatv2"
    layers: int = 3
    hidden: int = 256
    output: int = 128
    heads: int = 4
    first_hidden_multiplier: int = 2
    dropout: float = 0.3
    attention_dropout: float = 0.0
    attention_slope: float = 0.2
    add_self_loops: bool = True
    share_weights: bool = False
    concat: bool = True
    residual: bool = False
    norm: str = "graph_layer_norm+batch_norm"
    activation_slope: float = 0.01
    relation_edge_dim: Optional[int] = None

    @property
    def width(self) -> int:
        """The encoder's output width, as its kind's declaration states it
        (`EncoderDeclaration.width`; GATv2: `output * heads`). A kind this
        build declares no encoder for has none: refused as its build is."""
        return require_encoder_declaration(self.kind).width(self)

    def validate(self) -> None:
        if self.kind not in ENCODER_KINDS:
            raise SpecError(f"unknown encoder kind {self.kind!r}; one of {list(ENCODER_KINDS)}")
        for name in ("layers", "hidden", "output", "heads", "first_hidden_multiplier"):
            _positive_int(f"encoder.{name}", getattr(self, name))
        for name in ("dropout", "attention_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise SpecError(f"encoder.{name} must be a number in [0, 1)")
        if self.norm != "graph_layer_norm+batch_norm":
            raise SpecError(f"unknown encoder norm {self.norm!r}")
        # The kind's own rules (GATv2's: its widths are output x heads), from
        # its declaration; a kind this build does not declare is refused at build.
        declaration = encoder_declaration(self.kind)
        if declaration is not None:
            declaration.validate(self)
        if self.relation_edge_dim is not None:
            _positive_int("encoder.relation_edge_dim", self.relation_edge_dim)
            if declaration is None or not declaration.takes_relation_edge_dim:
                takes = [kind for kind, declared in ENCODER_DECLARATIONS.items()
                         if declared.takes_relation_edge_dim]
                raise SpecError(f"encoder.relation_edge_dim applies to "
                                f"{' and '.join(takes) or 'no encoder of this build'} only "
                                "(README §3.3 S3)")


#: The two computations an encoder can offer (README §2.8): `sampled`, a
#: sampled subgraph's seeds computed hop by hop (training's steps), and
#: `full_view`, every view node (validation, measurement and serving, through
#: `src.models.tasks.scoring.full_view_vectors`).
SAMPLED = "sampled"
FULL_VIEW = "full_view"


class EncoderDeclaration:
    """What one encoder kind's implementation declares, read without torch
    (README §3.3; research comparison §6 E2).

    **An encoder's own rules live in its declaration**, never in a shared
    path: the rules a spec of its kind obeys beyond the shared field checks
    (`validate`, called by `EncoderSpec.validate`), its output width
    (`width`), the computations it offers (`computation`), the sampler hops
    its sampled computation needs (`sampled_hops`, which `hops_problem` asks)
    and the inputs it reads besides each node's row (`reads_node_types`,
    `reads_edge_relations`). The capability description
    (`src.models.bundle.capabilities.model_capabilities`) and the start,
    training and page checks ask these and nothing else of a kind; how an
    encoder runs (GATv2's destination chunks, `chunk_size`) stays its
    implementation's execution parameter.

    `ENCODER_DECLARATIONS` holds a declaration for each kind this build
    implements; the one builder (`src.models.bundle.builder`) builds only a
    kind declared here that it has an implementation of, and refuses any
    other. A declaration is never a claim that a kind trains: only what is
    implemented and tested is declared.
    """

    #: The kind (`EncoderSpec.kind`) this declares.
    kind: str = ""
    #: The computations its implementation offers (`SAMPLED`, `FULL_VIEW`).
    computation: Tuple[str, ...] = ()
    #: What a sampler's hops are for, in the words `hops_problem` says.
    hops_words: str = ""
    #: Whether a spec of this kind may set `encoder.relation_edge_dim`.
    takes_relation_edge_dim: bool = False

    def validate(self, spec: "EncoderSpec") -> None:
        """Refuse a spec of this kind that breaks one of its own rules."""

    def width(self, spec: "EncoderSpec") -> int:
        """The output width of a spec of this kind: one vector of it per node."""
        raise NotImplementedError

    def sampled_hops(self, spec: "EncoderSpec") -> Optional[int]:
        """The hops its sampled computation needs (`num_neighbors` has one
        number per hop), or None when it offers no sampled computation."""
        raise NotImplementedError

    def reads_node_types(self, spec: "EncoderSpec") -> bool:
        """Whether a spec of this kind reads each node's type id."""
        return False

    def reads_edge_relations(self, spec: "EncoderSpec") -> bool:
        """Whether a spec of this kind reads each message-passing edge's relation id."""
        return False


class GATv2Declaration(EncoderDeclaration):
    """The reference encoder's declaration (README C1, C2, U1, U4, §2.8,
    §3.3 S3): homogeneous GATv2, every layer's heads concatenated, so its
    output width is `output * heads`; its sampled computation takes one hop
    per layer, its full-view computation runs layer by layer in destination
    chunks; it reads each message-passing edge's relation id when
    `relation_edge_dim` is set, and no node type."""

    kind = "gatv2"
    computation = (FULL_VIEW, SAMPLED)
    hops_words = "one hop per encoder layer"
    takes_relation_edge_dim = True

    def validate(self, spec: "EncoderSpec") -> None:
        if not spec.concat:
            raise SpecError("encoder.concat must be true: widths are output x heads (README C1)")

    def width(self, spec: "EncoderSpec") -> int:
        return spec.output * spec.heads

    def sampled_hops(self, spec: "EncoderSpec") -> Optional[int]:
        return spec.layers

    def reads_edge_relations(self, spec: "EncoderSpec") -> bool:
        return spec.relation_edge_dim is not None


#: The encoder kinds this build implements, each by its declaration. Read when
#: a judgment is made, never copied at import.
ENCODER_DECLARATIONS: Dict[str, EncoderDeclaration] = {"gatv2": GATv2Declaration()}


def encoder_declaration(kind: Any) -> Optional[EncoderDeclaration]:
    """The declaration of the encoder kind `kind`, or None when this build
    implements no encoder of that kind."""
    return ENCODER_DECLARATIONS.get(kind) if isinstance(kind, str) else None


def encoder_not_built(kind: Any) -> str:
    """Why an encoder of `kind` is not built: the one sentence of the builder
    and of every question asked of an encoder this build does not implement."""
    return (f"encoder kind {kind!r} is declared but not yet behind the GraphEncoder contract "
            "in this build")


def require_encoder_declaration(kind: Any) -> EncoderDeclaration:
    """The declaration of `kind`, or a `SpecError` saying it is not built."""
    declaration = encoder_declaration(kind)
    if declaration is None:
        raise SpecError(encoder_not_built(kind))
    return declaration


@dataclass(frozen=True)
class LPHeadSpec:
    """DistMult with tanh and a hinge loss (README A2)."""

    kind: str = "distmult_tanh"
    margin: float = 1.0
    relation_init: str = "xavier_uniform_leaky_relu"

    def validate(self) -> None:
        if self.kind != "distmult_tanh":
            raise SpecError(f"unknown LP head {self.kind!r}")
        if self.relation_init != "xavier_uniform_leaky_relu":
            raise SpecError(f"unknown relation init {self.relation_init!r}")
        if self.margin <= 0:
            raise SpecError("lp_head.margin must be positive")


@dataclass(frozen=True)
class PatientEncoderSpec:
    """Transformer over the patient's phenotype vectors, then attention
    pooling (README A6, U11). `layers` 0 is allowed, as in the paper's search."""

    transformer_layers: int = 3
    transformer_heads: int = 8
    feedforward: int = 2048
    dropout: float = 0.1

    def validate(self) -> None:
        if self.transformer_layers < 0:
            raise SpecError("patient_encoder.transformer_layers must be a non-negative integer")
        _positive_int("patient_encoder.transformer_heads", self.transformer_heads)
        _positive_int("patient_encoder.feedforward", self.feedforward)
        if not 0 <= self.dropout < 1:
            raise SpecError("patient_encoder.dropout must be in [0, 1)")


@dataclass(frozen=True)
class GeneScoringSpec:
    """The gene task's score (README §6, F10)."""

    eta: float = 0.5
    sp_lower: float = 1.0
    sp_upper: float = 6.84
    sp_aggregation: str = "mean"
    augmentation: bool = True
    augmentation_k: int = 3
    augmentation_omega: float = 0.7

    def validate(self) -> None:
        if not 0 <= self.eta <= 1:
            raise SpecError("task_head.gene.eta must be in [0, 1]")
        if not self.sp_upper > self.sp_lower:
            raise SpecError(f"task_head.gene SP bounds need upper > lower "
                            f"(got lower {self.sp_lower}, upper {self.sp_upper}); refused")
        if self.sp_aggregation != "mean":
            raise SpecError("task_head.gene.sp_aggregation must be 'mean'")
        _positive_int("task_head.gene.augmentation_k", self.augmentation_k)
        if not 0 <= self.augmentation_omega <= 1:
            raise SpecError("task_head.gene.augmentation_omega must be in [0, 1]")


@dataclass(frozen=True)
class TaskHeadSpec:
    """The task head of a finetuned bundle (README A7–A9, C5, C8, C9, U13).
    `multi_similarity` holds the loss's alpha, beta, gamma and margin."""

    task: str = "disease"
    projection_slope: float = 0.1
    distance_normalisation_eps: float = 1e-12
    softmax_scale: float = 1.0
    multi_similarity: Tuple[float, float, float, float] = (1.0, 20.0, 1.0, 0.4)
    gene: Optional[GeneScoringSpec] = None

    def validate(self) -> None:
        if self.task not in TASKS:
            raise SpecError(f"unknown task {self.task!r}; one of {list(TASKS)}")
        if (self.task == "gene") != (self.gene is not None):
            raise SpecError("task_head.gene is required for the gene task and only for it")
        if self.gene is not None:
            self.gene.validate()


@dataclass(frozen=True)
class ModelSpec:
    """A model bundle as one record.

    A base has no patient encoder and no task head; a finetuned bundle has both,
    and its graph view, input, encoder and LP head are its parent's (README
    §3.3, "Inheritance").
    """

    learning_stage: str
    graph_view: GraphViewSpec
    input: InputSpec
    encoder: EncoderSpec
    lp_head: LPHeadSpec
    patient_encoder: Optional[PatientEncoderSpec] = None
    task_head: Optional[TaskHeadSpec] = None
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def task(self) -> Optional[str]:
        return self.task_head.task if self.task_head is not None else None

    def validate(self) -> "ModelSpec":
        _require_declared_types(self)
        if self.learning_stage not in LEARNING_STAGES:
            raise SpecError(f"unknown learning stage {self.learning_stage!r}")
        self.graph_view.validate()
        self.input.validate()
        self.encoder.validate()
        self.lp_head.validate()
        if self.learning_stage == "base":
            if self.patient_encoder is not None or self.task_head is not None:
                raise SpecError("a base spec has no patient encoder and no task head")
        else:
            if self.patient_encoder is None or self.task_head is None:
                raise SpecError("a finetune spec needs a patient encoder and a task head")
            self.patient_encoder.validate()
            self.task_head.validate()
        return self

    def differences_from_reference(self) -> Dict[str, Tuple[Any, Any]]:
        """Each field that differs from the reference recipe, as
        `{"section.field": (reference, this)}`. Task heads are compared with
        the reference head of the same task."""
        reference = reference_spec(self.learning_stage, self.task)
        out: Dict[str, Tuple[Any, Any]] = {}
        mine, theirs = self.to_dict(), reference.to_dict()
        for section in ("graph_view", "input", "encoder", "lp_head", "patient_encoder",
                        "task_head"):
            _diff(section, theirs.get(section), mine.get(section), out)
        return out

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "spec_version": SPEC_VERSION,
            "learning_stage": self.learning_stage,
            "graph_view": _plain(asdict(self.graph_view)),
            "input": _plain(asdict(self.input)),
            "encoder": _plain(asdict(self.encoder)),
            "lp_head": _plain(asdict(self.lp_head)),
            "patient_encoder": (_plain(asdict(self.patient_encoder))
                                if self.patient_encoder is not None else None),
            "task_head": (_plain(asdict(self.task_head))
                          if self.task_head is not None else None),
            "sources": dict(self.sources),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        top = ("spec_version", "learning_stage", "graph_view", "input", "encoder", "lp_head",
               "patient_encoder", "task_head", "sources")
        _check_keys("spec", data, top, top)
        version = data["spec_version"]
        # An integer first (the type rule's sense: not a boolean), so a value
        # with no one answer to `!=` (a tensor) is refused, never compared.
        if not isinstance(version, int) or isinstance(version, bool) or version != SPEC_VERSION:
            raise SpecError(f"model spec version {_shown(version)} is not {SPEC_VERSION}; "
                            "this build reads only its own version")
        view = dict(_section("graph_view", data["graph_view"]))
        view["exclude_relations"] = _record_tuple(view.get("exclude_relations", ()))
        head = None
        if data["task_head"] is not None:
            head_data = dict(_section("task_head", data["task_head"]))
            gene = head_data.get("gene")
            head_data["gene"] = _from(GeneScoringSpec, "task_head.gene", gene) if gene is not None else None
            head_data["multi_similarity"] = _record_tuple(head_data.get("multi_similarity", ()))
            head = _from(TaskHeadSpec, "task_head", head_data)
        spec = cls(
            learning_stage=data["learning_stage"],
            graph_view=_from(GraphViewSpec, "graph_view", view),
            input=_from(InputSpec, "input", data["input"]),
            encoder=_from(EncoderSpec, "encoder", data["encoder"]),
            lp_head=_from(LPHeadSpec, "lp_head", data["lp_head"]),
            patient_encoder=(_from(PatientEncoderSpec, "patient_encoder", data["patient_encoder"])
                             if data["patient_encoder"] is not None else None),
            task_head=head,
            sources=(dict(data["sources"]) if isinstance(data["sources"], Mapping)
                     else data["sources"]),
        )
        return spec.validate()


@dataclass(frozen=True)
class ViewShape:
    """What the builder needs from a graph view, read from its record."""

    digest: str
    num_nodes: int
    node_types: Tuple[str, ...]
    relations: Tuple[str, ...]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ViewShape":
        """The shape of a graph view record (`record()`, `GraphView.record()`).

        **Each field is judged as the kind the view writes before anything is
        made of it**, nothing cast: the digest a non-empty string, the node
        count an integer (not a boolean) above zero, the node types and
        relations lists (or tuples) of names, at least one relation. So a value
        the safe loader reads but no view writes (an infinite count, which
        `int()` raised `OverflowError` on, a tensor, a list holding itself) is
        refused naming its field, never converted (the reviewer's K8-4r2
        recheck: `num_nodes: inf` made the whole model list answer HTTP 500)."""
        # A mapping first, so a value with no one answer to `in` (a tensor the
        # safe loader accepts) is refused, never asked.
        if not isinstance(record, Mapping):
            raise SpecError(f"the graph view record is {type(record).__name__}, not a mapping")
        missing = [k for k in ("digest", "num_nodes", "node_types", "relations") if k not in record]
        if missing:
            raise SpecError(f"the graph view record lacks {missing}")
        digest, num_nodes = record["digest"], record["num_nodes"]
        if not isinstance(digest, str) or not digest:
            raise SpecError(f"graph_view.digest must be a non-empty string, not {_shown(digest)}")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
            raise SpecError(f"graph_view.num_nodes must be an integer above zero, not "
                            f"{_shown(num_nodes)}")
        names = {}
        for key in ("node_types", "relations"):
            value = record[key]
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise SpecError(f"graph_view.{key} must be a list of names, not {_shown(value)}")
            names[key] = tuple(value)
        if not names["relations"]:
            raise SpecError("graph_view.relations names no relation")
        return cls(digest, num_nodes, names["node_types"], names["relations"])

    def record(self) -> Dict[str, Any]:
        return {"digest": self.digest, "num_nodes": self.num_nodes,
                "node_types": list(self.node_types), "relations": list(self.relations)}


def _positive_int(name: str, value: int) -> None:
    """An integer field's range (the type rule held it to an integer)."""
    if value <= 0:
        raise SpecError(f"{name} must be a positive integer (got {value!r})")


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _diff(prefix: str, reference: Any, mine: Any, out: Dict[str, Tuple[Any, Any]]) -> None:
    if isinstance(reference, dict) and isinstance(mine, dict):
        for key in sorted(set(reference) | set(mine)):
            _diff(f"{prefix}.{key}", reference.get(key), mine.get(key), out)
    elif reference != mine:
        out[prefix] = (reference, mine)


def finetune_head(task: str) -> Tuple[PatientEncoderSpec, TaskHeadSpec]:
    """The reference patient encoder and task head for `task` (README §2)."""
    if task not in TASKS:
        raise SpecError(f"unknown task {task!r}; one of {list(TASKS)}")
    gene = GeneScoringSpec() if task == "gene" else None
    return PatientEncoderSpec(), TaskHeadSpec(task=task, gene=gene)


def reference_spec(learning_stage: str = "base", task: Optional[str] = None) -> ModelSpec:
    """The reference recipe, anchored on the fixed source with the listed
    corrections (README §2). Every value is a provisional default (§0.3)."""
    if learning_stage == "base":
        if task is not None:
            raise SpecError("a base spec has no task")
        return ModelSpec("base", GraphViewSpec(), InputSpec(), EncoderSpec(), LPHeadSpec(),
                         sources=dict(REFERENCE_SOURCES)).validate()
    if task is None:
        raise SpecError("a finetune spec needs a task")
    patient, head = finetune_head(task)
    return ModelSpec("finetune", GraphViewSpec(), InputSpec(), EncoderSpec(), LPHeadSpec(),
                     patient, head, sources=dict(REFERENCE_SOURCES)).validate()


def hops_problem(num_neighbors: Sequence[Any], encoder: EncoderSpec) -> Optional[str]:
    """Why a sampler's hop list (`num_neighbors`) does not fit `encoder`, or
    None: the list has one number per hop the encoder's sampled computation
    needs, **as its kind's declaration states it**
    (`EncoderDeclaration.sampled_hops`; GATv2: one hop per layer). **The one
    statement of this check**: the start's check of a new base run
    (`training_manager._new_bundle_run`), the Fine-tune start's of each
    child against its base's encoder, before any child starts
    (`TrainingManager.start_finetune`), the Training page's base form
    (`training_console.setting_problem`) and the training process
    (`scripts/train_model.py`) all ask it. No hop list fits an encoder a run
    cannot sample, so the run is refused before it starts: one whose kind this
    build does not implement, in the builder's own sentence
    (`encoder_not_built`), and one whose declaration offers no sampled
    computation."""
    declaration = encoder_declaration(encoder.kind)
    if declaration is None:
        return f"{encoder_not_built(encoder.kind)}."
    hops = declaration.sampled_hops(encoder) if SAMPLED in declaration.computation else None
    if hops is None:
        return (f"encoder kind {encoder.kind!r} offers no sampled computation in this build, "
                "which training's sampler needs.")
    if len(num_neighbors) == hops:
        return None
    return (f"num_neighbors needs {hops} numbers, {declaration.hops_words}, "
            f"not {len(num_neighbors)}.")


#: The parts of a base's spec a finetune keeps (README §3.3, Inheritance):
#: its graph side, as `finetune_spec_from_parent` copies it.
FINETUNE_KEPT = ("graph_view", "input", "encoder", "lp_head")


def run_spec_problem(spec: ModelSpec, learning_stage: Optional[str], task: Optional[str], *,
                     base: Optional[ModelSpec] = None) -> Optional[str]:
    """Why `spec` cannot be the model spec of a run of `learning_stage` and
    `task`, or of a finetune of the base whose spec is `base`, or None. **The
    one check of a run's spec against its stage and its base**: the training
    process (`train_model.train`, every run), the job runner's new base run
    (`training_manager._new_bundle_run`) and its Fine-tune start of one task,
    before any child starts (`TrainingManager.start_finetune`), ask it alike.

    - a spec of another stage or task than the run's;
    - a finetune's spec whose graph side (`FINETUNE_KEPT`) is not its base's:
      a finetune never changes its base's graph view, input, encoder or LP
      head (README §3.3, Inheritance), only its patient encoder and task head
      are its own."""
    if spec.learning_stage != learning_stage or spec.task != task:
        def named(stage: Any, of: Any) -> str:
            return f"a {stage} spec" + (f" for {of}" if of is not None else "")

        return (f"model_spec is {named(spec.learning_stage, spec.task)}, not "
                f"{named(learning_stage, task)}")
    if base is not None and any(getattr(spec, part) != getattr(base, part)
                                for part in FINETUNE_KEPT):
        return ("a finetune keeps its base's graph view, input, encoder and LP head "
                "(README §3.3, Inheritance)")
    return None


def finetune_spec_from_parent(parent: ModelSpec, task: str) -> ModelSpec:
    """A finetune spec over `parent`'s graph side, with the reference head for
    `task`. A finetune never changes the parent's encoder (README §3.3)."""
    if parent.learning_stage != "base":
        raise SpecError("a finetune's parent is a base: children never chain")
    patient, head = finetune_head(task)
    return ModelSpec("finetune", parent.graph_view, parent.input, parent.encoder,
                     parent.lp_head, patient, head, sources=dict(parent.sources)).validate()


__all__ = [
    "ENCODER_DECLARATIONS",
    "ENCODER_KINDS",
    "EncoderDeclaration",
    "EncoderSpec",
    "FINETUNE_KEPT",
    "FULL_VIEW",
    "GATv2Declaration",
    "GeneScoringSpec",
    "GraphViewSpec",
    "InputSpec",
    "LEARNING_STAGES",
    "LPHeadSpec",
    "ModelSpec",
    "PatientEncoderSpec",
    "REFERENCE_SOURCES",
    "SAMPLED",
    "SPEC_VERSION",
    "SpecError",
    "TASKS",
    "TaskHeadSpec",
    "ViewShape",
    "declared_fields",
    "encoder_declaration",
    "encoder_not_built",
    "finetune_head",
    "finetune_spec_from_parent",
    "hops_problem",
    "reference_spec",
    "require_encoder_declaration",
    "run_spec_problem",
]
