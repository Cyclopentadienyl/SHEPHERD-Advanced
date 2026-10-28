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
a spec it cannot read completely is never half-read.

**Provisional defaults carry their source** (README §0.3). `REFERENCE_SOURCES`
names where each value of the reference recipe comes from; a spec built from
`reference_spec()` keeps that map in `sources`, and a value changed by an
operator is recorded as such by whoever changes it.

Module: src/models/bundle/spec.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Tuple

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


def _check_keys(section: str, data: Mapping[str, Any], allowed: Tuple[str, ...],
                required: Tuple[str, ...] = ()) -> None:
    if not isinstance(data, Mapping):
        raise SpecError(f"model spec section {section!r} is {type(data).__name__}, not a mapping")
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
        for flag in ("drop_native_self_loops", "deduplicate", "largest_component"):
            if not isinstance(getattr(self, flag), bool):
                raise SpecError(f"graph_view.{flag} must be a boolean")
        if any(not isinstance(r, str) or r.count("|") != 2 for r in self.exclude_relations):
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
        if not isinstance(self.type_embedding, bool):
            raise SpecError("input.type_embedding must be a boolean")


@dataclass(frozen=True)
class EncoderSpec:
    """The GraphEncoder (README C1, C2, U1, U3, U4, §2.8).

    For `gatv2` the layer widths follow the source: the first layer's hidden
    size is `hidden * first_hidden_multiplier`, the middle ones `hidden`, the
    last `output`; every layer has `heads` heads, concatenated, so the encoder's
    output width is `output * heads`. The other kinds are non-reference
    encoders behind the same contract: `hidden` is their width per layer and
    `output * heads` their output width too, so the LP head and task heads read
    one number.
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
        return int(self.output) * int(self.heads)

    def validate(self) -> None:
        if self.kind not in ENCODER_KINDS:
            raise SpecError(f"unknown encoder kind {self.kind!r}; one of {list(ENCODER_KINDS)}")
        for name in ("layers", "hidden", "output", "heads", "first_hidden_multiplier"):
            _positive_int(f"encoder.{name}", getattr(self, name))
        for name in ("dropout", "attention_dropout"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value < 1:
                raise SpecError(f"encoder.{name} must be a number in [0, 1)")
        if self.norm != "graph_layer_norm+batch_norm":
            raise SpecError(f"unknown encoder norm {self.norm!r}")
        if not self.concat:
            raise SpecError("encoder.concat must be true: widths are output x heads (README C1)")
        if self.relation_edge_dim is not None:
            _positive_int("encoder.relation_edge_dim", self.relation_edge_dim)
            if self.kind != "gatv2":
                raise SpecError("encoder.relation_edge_dim applies to gatv2 only (README §3.3 S3)")


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
        if not isinstance(self.margin, (int, float)) or self.margin <= 0:
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
        if not isinstance(self.transformer_layers, int) or self.transformer_layers < 0:
            raise SpecError("patient_encoder.transformer_layers must be a non-negative integer")
        _positive_int("patient_encoder.transformer_heads", self.transformer_heads)
        _positive_int("patient_encoder.feedforward", self.feedforward)
        if not 0 <= float(self.dropout) < 1:
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
        import math

        for name in ("eta", "sp_lower", "sp_upper", "augmentation_omega"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)):
                raise SpecError(f"task_head.gene.{name} must be a finite number")
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
    """The task head of a finetuned bundle (README A7–A9, C5, C8, C9, U13)."""

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
        if len(self.multi_similarity) != 4:
            raise SpecError("task_head.multi_similarity holds alpha, beta, gamma, margin")


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
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.sources.items()):
            raise SpecError("sources maps field names to source strings")
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
        if data["spec_version"] != SPEC_VERSION:
            raise SpecError(f"model spec version {data['spec_version']!r} is not {SPEC_VERSION}; "
                            "this build reads only its own version")
        view = dict(data["graph_view"])
        view["exclude_relations"] = tuple(view.get("exclude_relations", ()))
        head = None
        if data["task_head"] is not None:
            head_data = dict(data["task_head"])
            gene = head_data.get("gene")
            head_data["gene"] = _from(GeneScoringSpec, "task_head.gene", gene) if gene is not None else None
            head_data["multi_similarity"] = tuple(head_data.get("multi_similarity", ()))
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
            sources=dict(data["sources"]),
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
        missing = [k for k in ("digest", "num_nodes", "node_types", "relations") if k not in record]
        if missing:
            raise SpecError(f"the graph view record lacks {missing}")
        shape = cls(str(record["digest"]), int(record["num_nodes"]),
                    tuple(record["node_types"]), tuple(record["relations"]))
        if shape.num_nodes <= 0 or not shape.relations or not shape.digest:
            raise SpecError("the graph view record has no nodes, no relations or no digest")
        return shape

    def record(self) -> Dict[str, Any]:
        return {"digest": self.digest, "num_nodes": self.num_nodes,
                "node_types": list(self.node_types), "relations": list(self.relations)}


def _positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
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


def finetune_spec_from_parent(parent: ModelSpec, task: str) -> ModelSpec:
    """A finetune spec over `parent`'s graph side, with the reference head for
    `task`. A finetune never changes the parent's encoder (README §3.3)."""
    if parent.learning_stage != "base":
        raise SpecError("a finetune's parent is a base: children never chain")
    patient, head = finetune_head(task)
    return ModelSpec("finetune", parent.graph_view, parent.input, parent.encoder,
                     parent.lp_head, patient, head, sources=dict(parent.sources)).validate()


__all__ = [
    "ENCODER_KINDS",
    "EncoderSpec",
    "GeneScoringSpec",
    "GraphViewSpec",
    "InputSpec",
    "LEARNING_STAGES",
    "LPHeadSpec",
    "ModelSpec",
    "PatientEncoderSpec",
    "REFERENCE_SOURCES",
    "SPEC_VERSION",
    "SpecError",
    "TASKS",
    "TaskHeadSpec",
    "ViewShape",
    "finetune_head",
    "finetune_spec_from_parent",
    "reference_spec",
]
