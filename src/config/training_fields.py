"""
Training field-spec — single declarative source for the training-start parameter set.
=====================================================================================
B-lite (config-authority decision, docs/CONFIG_AUTHORITY.md). This module is the one
declarative description of the ~30 user-facing training-start parameters that today live,
hand-maintained and drifting, across three surfaces:

    - the API request model   (src/api/routes/training.py :: TrainingStartRequest)
    - the WebUI form + collect (src/webui/components/training_console.py)
    - the CLI + projection hub (scripts/train_model.py :: TrainConfig -> downstream configs)

Rollout phase (this file = PR-4a): **descriptive only.** It PINS current behavior; nothing
imports it yet and no surface derives from it.

What PR-4a's parity tests (tests/unit/test_training_fields.py) MECHANICALLY pin — so a drift
turns a test red — and what is deliberately deferred:

  Mechanically pinned now:
    - API: field existence, defaults, name coverage, AND the numeric constraints
      (``current_api`` = the exact ge/gt/le/lt dict, compared against the live Pydantic model).
    - WebUI collect: which fields ``_collect_config`` emits (``webui_exposed``), the default it
      produces for each, and the ``_num`` range guards it enforces.
    - Projection: every ``projects_to`` target attribute is checked to exist on the real config
      class (ShepherdGNNConfig / DataLoaderConfig / TrainerConfig / LossConfig / TrainConfig).
  Deliberately NOT pinned in PR-4a (narrowed claim):
    - The Slider/Dropdown widget min/max/choices live only on the gr components created inside
      ``create_training_tab()``; they are not introspectable without changing WebUI source, which
      PR-4a must not do. They are recorded here as ``ui`` / ``ui_choices`` (the source of truth) and
      become pinned-by-construction when PR-4d derives the widgets from this spec.

Later phases:
    - PR-4b: wire the three ``effective=False`` no-op fields through TrainConfig -> LossConfig.
    - PR-4c: enforce the ``valid`` / ``choices`` target policy on the API (+ CLI) and adopt the
      ``num_neighbors`` policy; WebUI keeps its conservative ``ui`` ranges.
    - PR-4d: derive WebUI widgets from this spec (removing SEED_PARAM_INDEX / positional coupling).

Two-bound model (approved): ``valid`` = hard validity per PyTorch/compute-lib norms (drives the
API in PR-4c); ``ui`` = conservative recommended range (drives the WebUI widget); ``ui`` ⊆ ``valid``.

Dependency-light (no torch / pydantic / gradio): any surface can read it and torch-free unit tests
can exercise it.

Module: src/config/training_fields.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

SCOPES = ("model", "dataloader", "trainer", "loss", "path", "runtime_setting")
KINDS = ("int", "float", "bool", "str", "list[int]")
# scopes whose effective fields are NOT required to project into TrainConfig:
_PROJECTION_EXEMPT_SCOPES = ("runtime_setting",)


@dataclass(frozen=True)
class FieldSpec:
    """One training-start parameter, described once.

    ``valid`` / ``choices`` / ``valid_pattern`` are the TARGET canonical policy (enforced in
    PR-4c). ``current_api`` is the STRUCTURED constraint the API enforces TODAY (dict of
    ge/gt/le/lt; ``{}`` = accepted but unconstrained, e.g. a free string; ``None`` = not an API
    field). ``webui_exposed`` records whether ``_collect_config`` emits the field today; ``ui`` /
    ``ui_choices`` / ``ui_widget`` are the conservative WebUI surface. Where current behavior
    differs from the target, ``known_divergence`` is True with a note.
    """

    name: str
    kind: str
    default: Any
    scope: str
    projects_to: Optional[str] = None            # "ShepherdGNNConfig.hidden_dim" etc.

    # target policy (PR-4c):
    valid: Optional[Tuple[Optional[float], Optional[float]]] = None
    valid_pattern: Optional[str] = None
    choices: Optional[Tuple[Any, ...]] = None

    # WebUI surface (structured):
    ui: Optional[Tuple[Optional[float], ...]] = None
    ui_choices: Optional[Tuple[Any, ...]] = None
    ui_widget: Optional[str] = None
    webui_exposed: bool = True                   # True => _collect_config emits it today

    # current API behavior pinned by PR-4a (structured):
    current_api: Optional[Dict[str, Any]] = None  # {} = unconstrained; None = not an API field

    current_webui: str = ""                       # human note; structure lives in ui*/webui_exposed
    known_divergence: bool = False
    divergence_note: str = ""

    effective: bool = True                        # False => accepted+validated but dropped at TrainConfig
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"{self.name}: unknown scope {self.scope!r}")
        if self.current_api is not None and not isinstance(self.current_api, dict):
            raise TypeError(f"{self.name}: current_api must be a dict or None")


# =============================================================================================
# The spec. current_api transcribed from src/api/routes/training.py::TrainingStartRequest;
# webui_exposed / ui from src/webui/components/training_console.py; scope/projects_to from the
# scripts/train_model.py projection into ShepherdGNNConfig/DataLoaderConfig/TrainerConfig/LossConfig.
# =============================================================================================
FIELDS: Tuple[FieldSpec, ...] = (
    # ---- paths (I/O; effective -> they DO project into TrainConfig) -------------------------
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
              valid=(32, None), ui_choices=(128, 256, 512), ui_widget="Dropdown",
              current_api={"ge": 32}, current_webui="Dropdown{128,256,512}",
              known_divergence=True,
              divergence_note="Asymmetric: API continuous >=32, WebUI 3 choices. Cross-field: "
                              "hidden_dim % num_heads == 0 for conv_type in {hgt,gat} (PR-4c, API-side).",
              description="Hidden dimension size."),
    FieldSpec("num_layers", "int", 4, "model", projects_to="ShepherdGNNConfig.num_layers",
              valid=(1, 16), ui=(2, 8, 1), ui_widget="Slider",
              current_api={"ge": 1, "le": 16}, current_webui="Slider 2-8",
              known_divergence=True, divergence_note="WebUI conservative 2-8 vs API 1-16.",
              description="Number of GNN layers."),
    FieldSpec("num_heads", "int", 8, "model", projects_to="ShepherdGNNConfig.num_heads",
              valid=(1, None), ui_choices=(4, 8, 16), ui_widget="Dropdown",
              current_api={"ge": 1}, current_webui="Dropdown{4,8,16}",
              known_divergence=True, divergence_note="Asymmetric. Used only when conv_type in {hgt,gat}.",
              description="Number of attention heads."),
    FieldSpec("dropout", "float", 0.1, "model", projects_to="ShepherdGNNConfig.dropout",
              valid=(0.0, 0.9), ui=(0.0, 0.5, 0.01), ui_widget="Slider",
              current_api={"ge": 0.0, "le": 0.9}, current_webui="Slider 0.0-0.5",
              known_divergence=True, divergence_note="WebUI caps at 0.5 vs API 0.9.",
              description="Dropout rate."),
    FieldSpec("use_ortholog_gate", "bool", True, "model", projects_to="ShepherdGNNConfig.use_ortholog_gate",
              current_api={}, current_webui="Checkbox", ui_widget="Checkbox",
              description="Use the cross-species ortholog gate."),

    # ---- dataloader (DataLoaderConfig) ------------------------------------------------------
    FieldSpec("batch_size", "int", 32, "dataloader", projects_to="DataLoaderConfig.batch_size",
              valid=(1, 2048), ui_choices=(8, 16, 32, 64, 128, 256, 512, 1024, 2048), ui_widget="Dropdown",
              current_api={"ge": 1, "le": 2048}, current_webui="Dropdown{8..2048}",
              known_divergence=True, divergence_note="API continuous 1-2048 vs WebUI discrete.",
              description="Batch size."),
    FieldSpec("num_neighbors", "list[int]", (15, 10, 5), "dataloader",
              projects_to="DataLoaderConfig.num_neighbors",
              current_api={}, current_webui="Textbox parsed; silent fallback [15,10,5] on error",
              known_divergence=True,
              divergence_note="PR-4c: API validates elements (int>=1, non-empty); WebUI toast-on-error "
                              "instead of silent fallback. len vs num_layers left unlinked (documented).",
              description="Neighbor fan-out per sampling hop."),
    FieldSpec("max_subgraph_nodes", "int", 5000, "dataloader", projects_to="DataLoaderConfig.max_subgraph_nodes",
              valid=(100, None), ui=(100, None, None), ui_widget="Number",
              current_api={"ge": 100}, current_webui="Number >=100 (_num)",
              description="Max nodes per sampled subgraph."),
    FieldSpec("num_workers", "int", 4, "dataloader", projects_to="DataLoaderConfig.num_workers",
              valid=(0, None), current_api={"ge": 0}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose num_workers.",
              description="DataLoader worker processes."),
    FieldSpec("num_negative_samples", "int", 5, "dataloader", projects_to="DataLoaderConfig.num_negative_samples",
              valid=(1, None), current_api={"ge": 1}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose num_negative_samples.",
              description="Negative samples per positive (train loader only)."),

    # ---- trainer (TrainerConfig) ------------------------------------------------------------
    FieldSpec("num_epochs", "int", 100, "trainer", projects_to="TrainerConfig.num_epochs",
              valid=(1, 10000), ui=(1, 10000, None), ui_widget="Number",
              current_api={"ge": 1, "le": 10000}, current_webui="Number 1-10000 (_num)",
              description="Number of training epochs."),
    FieldSpec("learning_rate", "float", 1e-4, "trainer", projects_to="TrainerConfig.learning_rate",
              valid=(0.0, 1.0), ui=(0.0, None, None), ui_widget="Number",
              current_api={"gt": 0, "le": 1.0}, current_webui="Number >0 (_num positive, no upper)",
              known_divergence=True, divergence_note="API caps at 1.0; WebUI unbounded above.",
              description="Learning rate."),
    FieldSpec("weight_decay", "float", 0.01, "trainer", projects_to="TrainerConfig.weight_decay",
              valid=(0.0, None), ui=(1e-5, 0.1, 1e-5), ui_widget="Slider",
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
              valid=(0, None), ui=(0, None, None), ui_widget="Number",
              current_api={"ge": 0}, current_webui="Number >=0 (_num)",
              description="Scheduler warmup steps."),
    FieldSpec("min_lr_ratio", "float", 0.01, "trainer", projects_to="TrainerConfig.min_lr_ratio",
              valid=(0.0, 1.0), ui=(1e-4, 1.0, None), ui_widget="Number",
              current_api={"ge": 0.0}, current_webui="Number 1e-4-1.0 (_num)",
              known_divergence=True,
              divergence_note="API allows 0 (decay-to-zero) & >1; WebUI clamps 1e-4..1.0. onecycle "
                              "requires >0 on both (API model_validator).",
              description="Final LR as a fraction of peak."),
    FieldSpec("gradient_accumulation_steps", "int", 1, "trainer",
              projects_to="TrainerConfig.gradient_accumulation_steps",
              valid=(1, None), ui=(1, None, None), ui_widget="Number",
              current_api={"ge": 1}, current_webui="Number >=1 (_num)",
              description="Micro-batches accumulated per optimizer step."),
    FieldSpec("max_grad_norm", "float", 1.0, "trainer", projects_to="TrainerConfig.max_grad_norm",
              valid=(0.0, None), ui=(0.01, None, None), ui_widget="Number",
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
              valid=(1, None), current_api={"ge": 1}, current_webui="not-exposed", webui_exposed=False,
              known_divergence=True, divergence_note="WebUI does not expose eval_every_n_epochs.",
              description="Run validation every N epochs."),
    FieldSpec("early_stopping_patience", "int", 10, "trainer", projects_to="TrainerConfig.early_stopping_patience",
              valid=(1, None), ui=(1, None, None), ui_widget="Number",
              current_api={"ge": 1}, current_webui="Number >=1 (_num)",
              description="Epochs without improvement before stopping."),
    FieldSpec("save_top_k", "int", 3, "trainer", projects_to="TrainerConfig.save_top_k",
              valid=(1, None), current_api={"ge": 1}, current_webui="not-exposed", webui_exposed=False,
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
              valid=(0, 2**32 - 1), ui=(0, 2**32 - 1, None), ui_widget="Number",
              current_api={"ge": 0, "le": 2**32 - 1}, current_webui="Number 0-4294967295 (_num)",
              description="Global RNG seed (numpy legal range)."),

    # ---- loss weights (LossConfig — projected) ---------------------------------------------
    FieldSpec("diagnosis_weight", "float", 1.0, "loss", projects_to="LossConfig.diagnosis_weight",
              valid=(0.0, None), ui=(0.0, 2.0, 0.1), ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Diagnosis-task loss weight."),
    FieldSpec("link_prediction_weight", "float", 0.5, "loss", projects_to="LossConfig.link_prediction_weight",
              valid=(0.0, None), ui=(0.0, 2.0, 0.1), ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Link-prediction loss weight."),
    FieldSpec("contrastive_weight", "float", 0.3, "loss", projects_to="LossConfig.contrastive_weight",
              valid=(0.0, None), ui=(0.0, 2.0, 0.1), ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Contrastive loss weight."),
    FieldSpec("ortholog_weight", "float", 0.2, "loss", projects_to="LossConfig.ortholog_weight",
              valid=(0.0, None), ui=(0.0, 2.0, 0.1), ui_widget="Slider",
              current_api={"ge": 0.0}, current_webui="Slider 0.0-2.0",
              known_divergence=True, divergence_note="WebUI caps at 2.0.",
              description="Ortholog loss weight."),

    # ---- loss knobs that are ACCEPTED-BUT-NOT-EFFECTIVE (no-op; PR-4b wires them) ------------
    FieldSpec("temperature", "float", 0.07, "loss", projects_to=None,
              valid=(0.0, None), ui=(0.01, 1.0, 0.01), ui_widget="Slider",
              current_api={"gt": 0.0}, current_webui="Slider 0.01-1.0",
              effective=False, known_divergence=True,
              divergence_note="No-op today: dropped by TrainConfig.hasattr gate; LossConfig default 0.07 "
                              "always used. PR-4b (R1) wires it through -> effective=True.",
              description="Contrastive temperature (LossConfig default 0.07 until PR-4b)."),
    FieldSpec("label_smoothing", "float", 0.1, "loss", projects_to=None,
              valid=(0.0, 1.0), ui=(0.0, 0.3, 0.01), ui_widget="Slider",
              current_api={"ge": 0.0, "le": 1.0}, current_webui="Slider 0.0-0.3",
              effective=False, known_divergence=True,
              divergence_note="No-op today: dropped by TrainConfig.hasattr gate; default 0.1 used. PR-4b wires it.",
              description="Diagnosis-loss label smoothing (LossConfig default 0.1 until PR-4b)."),
    FieldSpec("margin", "float", 1.0, "loss", projects_to=None,
              valid=(0.0, None), ui=(0.1, 3.0, 0.1), ui_widget="Slider",
              current_api={"gt": 0.0}, current_webui="Slider 0.1-3.0",
              effective=False, known_divergence=True,
              divergence_note="No-op today: dropped by TrainConfig.hasattr gate; default 1.0 used. PR-4b wires it.",
              description="Ranking-loss margin (LossConfig default 1.0 until PR-4b)."),

    # ---- runtime setting (from Runtime Settings file; exempt from the projection requirement) -
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
