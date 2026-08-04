"""
Training field-spec — single declarative source for the training-start parameter set.
=====================================================================================
B-lite (config-authority decision, docs/CONFIG_AUTHORITY.md). This module is the one
declarative description of the ~30 user-facing training-start parameters that today live,
hand-maintained and drifting, across three surfaces:

    - the API request model   (src/api/routes/training.py :: TrainingStartRequest)
    - the WebUI form + collect (src/webui/components/training_console.py)
    - the CLI + projection hub (scripts/train_model.py :: TrainConfig -> downstream configs)

Rollout phase (this file = PR-4a): **descriptive only.** It PINS current behavior and REPRESENTS
the target policy; nothing imports it yet and no surface derives from it.

Constraint representation
-------------------------
All bounds are structured dicts using the Pydantic/annotated-types vocabulary
``{"ge"|"gt"|"le"|"lt": value}`` so inclusive and exclusive bounds are distinguishable:

    valid       — TARGET hard validity (PyTorch/compute-lib norms). Drives the API in PR-4c.
    ui          — CURRENT conservative WebUI bounds. ``ui`` ⊆ ``valid`` except where a bound key
                  is declared in ``ui_wider_than_valid`` (a real, self-validated divergence).
    current_api — CURRENT API constraints, compared mechanically against the live Pydantic model.

``valid`` and ``current_api`` share one shape on purpose: PR-4c's job is exactly the diff between
them, and cross-field rules live in ``CROSS_FIELD_RULES`` rather than in prose, so PR-4c consumes
(or is guarded against) the spec instead of re-deriving policy.

What PR-4a's parity tests mechanically pin — and what is deliberately deferred:

  Pinned now:
    - API: field existence, defaults, name coverage, and the exact ge/gt/le/lt constraints.
    - WebUI collect: which fields ``_collect_config`` emits (``webui_exposed``), each default, and
      EVERY ``_num`` range guard (``webui_num_guarded``), derived from ``ui`` — both sides where
      the field is bounded on both sides.
    - Projection: every ``projects_to`` target attribute exists on the real config class.
    - Self-validation: constraint dicts are well-formed; cross-field rules reference real fields
      and real choices; declared ui/valid divergences are actually divergent.
  Deliberately NOT pinned in PR-4a (narrowed claim):
    - Slider/Dropdown widget min/max/choices live only on the gr components created inside
      ``create_training_tab()``; introspecting them would require changing WebUI source, which
      PR-4a must not do. They are recorded here as ``ui`` / ``ui_choices`` / ``ui_step`` (the
      source of truth) and become pinned-by-construction when PR-4d derives the widgets.

Phase status:
    - PR-4b (done): ``temperature`` / ``label_smoothing`` / ``margin`` are wired through
      ``TrainConfig`` into ``LossConfig``. They were previously accepted and validated by both
      surfaces but silently dropped by ``load_config``'s ``hasattr`` gate, so LossConfig always
      used its own defaults. The TrainConfig defaults added there match LossConfig exactly, so
      runs that do not set them are unchanged; runs that do now get the advertised effect.
      No field is ``effective=False`` any more.

Later phases:
    - PR-4c: enforce ``valid`` / ``choices`` / ``item_valid`` / ``CROSS_FIELD_RULES`` on the API
      (+ relax the CLI ``--device`` choices); WebUI keeps its conservative ``ui`` ranges.
    - PR-4d: derive WebUI widgets from this spec (removing SEED_PARAM_INDEX / positional coupling).

Dependency-light (no torch / pydantic / gradio): any surface can read it and torch-free unit tests
can exercise it.

Module: src/config/training_fields.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

SCOPES = ("model", "dataloader", "trainer", "loss", "path", "runtime_setting")
KINDS = ("int", "float", "bool", "str", "list[int]")
BOUND_KEYS = ("ge", "gt", "le", "lt")
# scopes whose effective fields are NOT required to project into TrainConfig:
_PROJECTION_EXEMPT_SCOPES = ("runtime_setting",)


@dataclass(frozen=True)
class CrossFieldRule:
    """A constraint spanning more than one field, represented structurally.

    ``when`` is an optional applicability guard: ``{"conv_type": ("hgt", "gat")}`` means the rule
    applies only when ``conv_type`` holds one of those values.
    """

    name: str
    kind: str                                            # "divisible_by" | "min_exclusive"
    subject: str                                         # field the rule constrains
    operand: Optional[str] = None                        # other field involved (divisible_by)
    threshold: Optional[float] = None                    # bound value (min_exclusive)
    when: Optional[Dict[str, Tuple[Any, ...]]] = None    # applicability guard
    enforced_in: str = ""                                # where it is / will be enforced
    description: str = ""


CROSS_FIELD_RULES: Tuple[CrossFieldRule, ...] = (
    CrossFieldRule(
        name="hidden_dim_divisible_by_num_heads",
        kind="divisible_by",
        subject="hidden_dim",
        operand="num_heads",
        when={"conv_type": ("hgt", "gat")},
        enforced_in="PR-4c (API model_validator)",
        description=(
            "hidden_dim % num_heads == 0. Required for hgt (HGTConv splits out_channels across "
            "heads) and gat (GATConv uses hidden_dim // num_heads with concat=True, so a "
            "non-divisible value truncates and breaks the residual add / LayerNorm(hidden_dim)). "
            "NOT applicable to sage: SAGEConv takes no heads."
        ),
    ),
    CrossFieldRule(
        name="onecycle_requires_positive_min_lr_ratio",
        kind="min_exclusive",
        subject="min_lr_ratio",
        threshold=0.0,
        when={"scheduler_type": ("onecycle",)},
        enforced_in="current (API _validate_onecycle_min_lr; trainer _validate_min_lr_ratio)",
        description=(
            "onecycle computes final_div_factor = 1 / min_lr_ratio, so it needs a strictly "
            "positive value; cosine/linear legitimately allow 0 (decay to zero)."
        ),
    ),
)


@dataclass(frozen=True)
class FieldSpec:
    """One training-start parameter, described once.

    ``valid`` / ``choices`` / ``valid_pattern`` / ``item_valid`` / ``min_length`` are the TARGET
    canonical policy (enforced in PR-4c). ``current_api`` is the STRUCTURED constraint the API
    enforces TODAY (``{}`` = accepted but unconstrained, e.g. a free string; ``None`` = not an API
    field). ``webui_exposed`` / ``webui_num_guarded`` / ``ui`` describe what ``_collect_config``
    does today.
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

    # WebUI surface (structured):
    ui: Optional[Dict[str, Any]] = None
    ui_step: Optional[float] = None
    ui_choices: Optional[Tuple[Any, ...]] = None
    ui_widget: Optional[str] = None
    webui_exposed: bool = True                    # _collect_config emits it today
    webui_num_guarded: bool = False               # _collect_config routes it through _num
    ui_wider_than_valid: Tuple[str, ...] = ()     # bound keys where current UI is looser than valid

    # current API behavior pinned by PR-4a:
    current_api: Optional[Dict[str, Any]] = None

    current_webui: str = ""
    known_divergence: bool = False
    divergence_note: str = ""

    effective: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"{self.name}: unknown scope {self.scope!r}")
        for label in ("valid", "ui", "current_api", "item_valid"):
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
# The spec. current_api transcribed from src/api/routes/training.py::TrainingStartRequest;
# ui / webui_* from src/webui/components/training_console.py::_collect_config; scope/projects_to
# from the scripts/train_model.py projection into the downstream configs.
# =============================================================================================
FIELDS: Tuple[FieldSpec, ...] = (
    # ---- paths (effective -> they DO project into TrainConfig) ------------------------------
    FieldSpec("data_dir", "str", "data/workspaces/default", "path", projects_to="TrainConfig.data_dir",
              current_api={}, current_webui="Textbox", ui_widget="Textbox",
              description="Workspace directory."),
    FieldSpec("output_dir", "str", "outputs", "path", projects_to="TrainConfig.output_dir",
              current_api={}, current_webui="Textbox", ui_widget="Textbox",
              description="Output directory."),
    FieldSpec("checkpoint_dir", "str", "", "path", projects_to="TrainConfig.checkpoint_dir",
              current_api={}, current_webui="Textbox", ui_widget="Textbox",
              description="Checkpoint directory (blank = auto-derive from workspace)."),
    FieldSpec("log_dir", "str", "logs", "path", projects_to="TrainConfig.log_dir",
              current_api={}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose log_dir.",
              description="Log directory."),
    FieldSpec("resume_from", "str", None, "path", projects_to="TrainConfig.resume_from",
              current_api={}, current_webui="not-exposed (WebUI uses a checkpoint dropdown)",
              webui_exposed=False, known_divergence=True,
              divergence_note="WebUI resume uses a checkpoint dropdown, not this field.",
              description="Checkpoint path to resume from."),

    # ---- model (ShepherdGNNConfig) ----------------------------------------------------------
    FieldSpec("conv_type", "str", "gat", "model", projects_to="ShepherdGNNConfig.conv_type",
              choices=("gat", "hgt", "sage"), ui_choices=("gat", "hgt", "sage"), ui_widget="Radio",
              current_api={}, current_webui="Radio{gat,hgt,sage}",
              known_divergence=True, divergence_note="API accepts any string today; enum enforced in PR-4c.",
              description="GNN convolution type."),
    FieldSpec("hidden_dim", "int", 256, "model", projects_to="ShepherdGNNConfig.hidden_dim",
              valid={"ge": 32}, ui_choices=(128, 256, 512), ui_widget="Dropdown",
              current_api={"ge": 32}, current_webui="Dropdown{128,256,512}",
              known_divergence=True,
              divergence_note="Asymmetric: API continuous >=32, WebUI 3 choices. See "
                              "CROSS_FIELD_RULES.hidden_dim_divisible_by_num_heads.",
              description="Hidden dimension size."),
    FieldSpec("num_layers", "int", 4, "model", projects_to="ShepherdGNNConfig.num_layers",
              valid={"ge": 1, "le": 16}, ui={"ge": 2, "le": 8}, ui_step=1, ui_widget="Slider",
              current_api={"ge": 1, "le": 16}, current_webui="Slider 2-8",
              known_divergence=True, divergence_note="WebUI conservative 2-8 vs API 1-16.",
              description="Number of GNN layers."),
    FieldSpec("num_heads", "int", 8, "model", projects_to="ShepherdGNNConfig.num_heads",
              valid={"ge": 1}, ui_choices=(4, 8, 16), ui_widget="Dropdown",
              current_api={"ge": 1}, current_webui="Dropdown{4,8,16}",
              known_divergence=True, divergence_note="Asymmetric. Used only when conv_type in {hgt,gat}.",
              description="Number of attention heads."),
    FieldSpec("dropout", "float", 0.1, "model", projects_to="ShepherdGNNConfig.dropout",
              valid={"ge": 0.0, "le": 0.9}, ui={"ge": 0.0, "le": 0.5}, ui_step=0.01, ui_widget="Slider",
              current_api={"ge": 0.0, "le": 0.9}, current_webui="Slider 0.0-0.5",
              known_divergence=True, divergence_note="WebUI caps at 0.5 vs API 0.9.",
              description="Dropout rate."),
    FieldSpec("use_ortholog_gate", "bool", True, "model", projects_to="ShepherdGNNConfig.use_ortholog_gate",
              current_api={}, current_webui="Checkbox", ui_widget="Checkbox",
              description="Use the cross-species ortholog gate."),

    # ---- dataloader (DataLoaderConfig) ------------------------------------------------------
    FieldSpec("batch_size", "int", 32, "dataloader", projects_to="DataLoaderConfig.batch_size",
              valid={"ge": 1, "le": 2048},
              ui_choices=(8, 16, 32, 64, 128, 256, 512, 1024, 2048), ui_widget="Dropdown",
              current_api={"ge": 1, "le": 2048}, current_webui="Dropdown{8..2048}",
              known_divergence=True, divergence_note="API continuous 1-2048 vs WebUI discrete.",
              description="Batch size."),
    FieldSpec("num_neighbors", "list[int]", (15, 10, 5), "dataloader",
              projects_to="DataLoaderConfig.num_neighbors",
              item_valid={"ge": 1}, min_length=1,
              current_api={}, current_webui="Textbox parsed; silent fallback [15,10,5] on error",
              known_divergence=True,
              divergence_note="Target (PR-4c): API validates non-empty and each element >=1 "
                              "(item_valid/min_length); WebUI toast-on-error instead of silent "
                              "fallback. len vs num_layers deliberately left unlinked.",
              description="Neighbor fan-out per sampling hop."),
    FieldSpec("max_subgraph_nodes", "int", 5000, "dataloader", projects_to="DataLoaderConfig.max_subgraph_nodes",
              valid={"ge": 100}, ui={"ge": 100}, ui_widget="Number", webui_num_guarded=True,
              current_api={"ge": 100}, current_webui="Number >=100 (_num)",
              description="Max nodes per sampled subgraph."),
    FieldSpec("num_workers", "int", 4, "dataloader", projects_to="DataLoaderConfig.num_workers",
              valid={"ge": 0}, current_api={"ge": 0}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose num_workers.",
              description="DataLoader worker processes."),
    FieldSpec("num_negative_samples", "int", 5, "dataloader", projects_to="DataLoaderConfig.num_negative_samples",
              valid={"ge": 1}, current_api={"ge": 1}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose num_negative_samples.",
              description="Negative samples per positive (train loader only)."),

    # ---- trainer (TrainerConfig) ------------------------------------------------------------
    FieldSpec("num_epochs", "int", 100, "trainer", projects_to="TrainerConfig.num_epochs",
              valid={"ge": 1, "le": 10000}, ui={"ge": 1, "le": 10000}, ui_widget="Number",
              webui_num_guarded=True,
              current_api={"ge": 1, "le": 10000}, current_webui="Number 1-10000 (_num)",
              description="Number of training epochs."),
    FieldSpec("learning_rate", "float", 1e-4, "trainer", projects_to="TrainerConfig.learning_rate",
              valid={"gt": 0, "le": 1.0}, ui={"gt": 0}, ui_widget="Number",
              webui_num_guarded=True, ui_wider_than_valid=("le",),
              current_api={"gt": 0, "le": 1.0}, current_webui="Number >0 (_num positive, no upper)",
              known_divergence=True,
              divergence_note="API caps at 1.0; WebUI has no upper bound (declared in "
                              "ui_wider_than_valid). PR-4d would give the widget the valid upper.",
              description="Learning rate."),
    FieldSpec("weight_decay", "float", 0.01, "trainer", projects_to="TrainerConfig.weight_decay",
              valid={"ge": 0.0}, ui={"ge": 1e-5, "le": 0.1}, ui_step=1e-5, ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 1e-5-0.1",
              known_divergence=True, divergence_note="API unbounded above; WebUI 1e-5..0.1.",
              description="Optimizer weight decay."),
    FieldSpec("scheduler_type", "str", "cosine", "trainer", projects_to="TrainerConfig.scheduler_type",
              choices=("cosine", "onecycle", "linear", "none"),
              ui_choices=("cosine", "onecycle", "linear", "none"), ui_widget="Dropdown",
              current_api={}, current_webui="Dropdown{cosine,onecycle,linear,none}",
              known_divergence=True, divergence_note="API accepts any string today; enum enforced in PR-4c.",
              description="LR scheduler type."),
    FieldSpec("warmup_steps", "int", 500, "trainer", projects_to="TrainerConfig.warmup_steps",
              valid={"ge": 0}, ui={"ge": 0}, ui_widget="Number", webui_num_guarded=True,
              current_api={"ge": 0}, current_webui="Number >=0 (_num)",
              description="Scheduler warmup steps."),
    FieldSpec("min_lr_ratio", "float", 0.01, "trainer", projects_to="TrainerConfig.min_lr_ratio",
              valid={"ge": 0.0, "le": 1.0}, ui={"ge": 1e-4, "le": 1.0}, ui_widget="Number",
              webui_num_guarded=True,
              current_api={"ge": 0.0}, current_webui="Number 1e-4-1.0 (_num)",
              known_divergence=True,
              divergence_note="API allows 0 (decay-to-zero) and has no upper; WebUI clamps "
                              "1e-4..1.0. onecycle needs >0 — see CROSS_FIELD_RULES.",
              description="Final LR as a fraction of peak."),
    FieldSpec("gradient_accumulation_steps", "int", 1, "trainer",
              projects_to="TrainerConfig.gradient_accumulation_steps",
              valid={"ge": 1}, ui={"ge": 1}, ui_widget="Number", webui_num_guarded=True,
              current_api={"ge": 1}, current_webui="Number >=1 (_num)",
              description="Micro-batches accumulated per optimizer step."),
    FieldSpec("max_grad_norm", "float", 1.0, "trainer", projects_to="TrainerConfig.max_grad_norm",
              valid={"gt": 0}, ui={"ge": 0.01}, ui_widget="Number", webui_num_guarded=True,
              current_api={"gt": 0.0}, current_webui="Number >=0.01 (_num)",
              known_divergence=True, divergence_note="API >0; WebUI >=0.01.",
              description="Gradient-norm clip threshold."),
    FieldSpec("use_amp", "bool", True, "trainer", projects_to="TrainerConfig.use_amp",
              current_api={}, current_webui="derived from amp_mode (+ hgt->off)",
              known_divergence=True, divergence_note="WebUI derives use_amp from the amp_mode Radio; composite.",
              description="Enable automatic mixed precision."),
    FieldSpec("amp_dtype", "str", "float16", "trainer", projects_to="TrainerConfig.amp_dtype",
              choices=("float16", "bfloat16"), ui_choices=("Off", "float16", "bfloat16"), ui_widget="Radio",
              current_api={}, current_webui="derived from amp_mode Radio{Off,float16,bfloat16}",
              known_divergence=True,
              divergence_note="API accepts any string (enum in PR-4c); WebUI models it as the amp_mode composite.",
              description="AMP compute dtype."),
    FieldSpec("eval_every_n_epochs", "int", 1, "trainer", projects_to="TrainerConfig.eval_every_n_epochs",
              valid={"ge": 1}, current_api={"ge": 1}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose eval_every_n_epochs.",
              description="Run validation every N epochs."),
    FieldSpec("early_stopping_patience", "int", 10, "trainer", projects_to="TrainerConfig.early_stopping_patience",
              valid={"ge": 1}, ui={"ge": 1}, ui_widget="Number", webui_num_guarded=True,
              current_api={"ge": 1}, current_webui="Number >=1 (_num)",
              description="Epochs without improvement before stopping."),
    FieldSpec("save_top_k", "int", 3, "trainer", projects_to="TrainerConfig.save_top_k",
              valid={"ge": 1}, current_api={"ge": 1}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose save_top_k.",
              description="Keep the best K checkpoints."),
    FieldSpec("device", "str", "auto", "trainer", projects_to="TrainerConfig.device",
              valid_pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$",
              ui_choices=("auto", "cuda", "cpu"), ui_widget="Radio",
              current_api={}, current_webui="Radio{auto,cuda,cpu}",
              known_divergence=True,
              divergence_note="NOT a closed enum: PR-4c validates PyTorch device grammar (cuda:N, mps); "
                              "WebUI keeps {auto,cuda,cpu}. CLI --device choices to be relaxed too.",
              description="Compute device."),
    FieldSpec("seed", "int", 42, "trainer", projects_to="TrainerConfig.seed",
              valid={"ge": 0, "le": 2**32 - 1}, ui={"ge": 0, "le": 2**32 - 1}, ui_widget="Number",
              webui_num_guarded=True,
              current_api={"ge": 0, "le": 2**32 - 1}, current_webui="Number 0-4294967295 (_num)",
              description="Global RNG seed (numpy legal range)."),

    # ---- loss weights (LossConfig — projected) ---------------------------------------------
    FieldSpec("diagnosis_weight", "float", 1.0, "loss", projects_to="LossConfig.diagnosis_weight",
              valid={"ge": 0.0}, ui={"ge": 0.0, "le": 2.0}, ui_step=0.1, ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Diagnosis-task loss weight."),
    FieldSpec("link_prediction_weight", "float", 0.5, "loss", projects_to="LossConfig.link_prediction_weight",
              valid={"ge": 0.0}, ui={"ge": 0.0, "le": 2.0}, ui_step=0.1, ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Link-prediction loss weight."),
    FieldSpec("contrastive_weight", "float", 0.3, "loss", projects_to="LossConfig.contrastive_weight",
              valid={"ge": 0.0}, ui={"ge": 0.0, "le": 2.0}, ui_step=0.1, ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Contrastive loss weight."),
    FieldSpec("ortholog_weight", "float", 0.2, "loss", projects_to="LossConfig.ortholog_weight",
              valid={"ge": 0.0}, ui={"ge": 0.0, "le": 2.0}, ui_step=0.1, ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Ortholog loss weight."),

    # ---- loss shape knobs (wired through TrainConfig -> LossConfig in PR-4b) -----------------
    FieldSpec("temperature", "float", 0.07, "loss", projects_to="LossConfig.temperature",
              valid={"gt": 0}, ui={"ge": 0.01, "le": 1.0}, ui_step=0.01, ui_widget="Slider",
              current_api={"gt": 0.0}, current_webui="Slider 0.01-1.0",
              known_divergence=True,
              divergence_note="WebUI caps at 1.0; API unbounded above.",
              description="Contrastive temperature."),
    FieldSpec("label_smoothing", "float", 0.1, "loss", projects_to="LossConfig.label_smoothing",
              valid={"ge": 0.0, "le": 1.0}, ui={"ge": 0.0, "le": 0.3}, ui_step=0.01, ui_widget="Slider",
              current_api={"ge": 0.0, "le": 1.0}, current_webui="Slider 0.0-0.3",
              known_divergence=True,
              divergence_note="WebUI caps at 0.3 vs API 1.0.",
              description="Diagnosis-loss label smoothing."),
    FieldSpec("margin", "float", 1.0, "loss", projects_to="LossConfig.margin",
              valid={"gt": 0}, ui={"ge": 0.1, "le": 3.0}, ui_step=0.1, ui_widget="Slider",
              current_api={"gt": 0.0}, current_webui="Slider 0.1-3.0",
              known_divergence=True,
              divergence_note="WebUI clamps to 0.1-3.0; API unbounded above.",
              description="Ranking-loss margin."),

    # ---- runtime setting (exempt from the projection requirement) ----------------------------
    FieldSpec("compile", "bool", False, "runtime_setting", projects_to="TrainConfig.compile",
              current_api=None, current_webui="derived from .shepherd_runtime_settings.json (torch_compile)",
              known_divergence=True,
              divergence_note="Runtime setting, not a training-form field. API does not expose it; "
                              "_collect_config emits it, sourced from the Runtime Settings file.",
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
