# Config Authority — Decision Record

**Status:** Accepted (2026-07). Supersedes the unused `HyperparameterManager`.

## Decision — Model B + B-lite

**Runtime configuration authority stays with the per-concern dataclasses:**

| Concern | Authority | Location |
|---|---|---|
| Training runtime | `TrainerConfig` | `src/training/trainer.py` |
| Inference runtime | `PipelineConfig` | `src/inference/pipeline.py` |
| Model architecture | `ModelSpec` (the model spec a bundle checkpoint records); the reference recipe for lifecycle defaults | `src/models/bundle/spec.py`; `src/config/recipe.py` |

*Model architecture (corrected at K7, 2026-10-29):* the model spec has been the architecture's
authority since K1: every model bundle is rebuilt from the spec its checkpoint records, through
the one builder. `ShepherdGNNConfig` (`src/models/gnn/shepherd_gnn.py`) describes only the
ShepherdGNN path, which stays for history files until K8. *(K8, 2026-11-08: that path is gone.
Its settings left `TrainConfig` in K8-7, and `ShepherdGNNConfig` with
`src/models/gnn/shepherd_gnn.py` in K8-10; the model spec is the architecture's only authority.)*

- **No central `HyperparameterManager`.** `src/config/hyperparameters.py` is **removed**: it
  was never on the live runtime path (only re-exported from `config/__init__.py` and exercised
  by a smoke test), and its spec had drifted from the dataclasses actually used — a second,
  unsynchronised source of truth.
- **B-lite (later PR):** the one genuinely shared config concern is the ~50-field training
  parameter set duplicated across the API `TrainingStartRequest`, the WebUI
  `_collect_config`/widgets (and the brittle positional `SEED_PARAM_INDEX`), and `TrainerConfig`.
  It will be consolidated into a **lightweight training field-spec** module
  (names / types / defaults / ranges / descriptions) that the API and WebUI *derive* from —
  **not** by resurrecting a heavyweight manager.
  - *Note (2026-10-28, K6):* the WebUI form this paragraph names is gone. `_collect_config`, its
    widgets and `SEED_PARAM_INDEX` were deleted with the ShepherdGNN form; the Training page's base
    form offers the reference recipe's base settings (`src/config/recipe.py` `BASE`), each checked
    by the field spec's `value_problem` (`src/config/training_fields.py`), the check the job
    runner's start makes again. The record above is left as written.
  - *Note (2026-11-09):* PR-4 is partly done (4a–4c, see Rollout). `src/config/training_fields.py`
    is the one lightweight training field-spec (names / types / defaults / ranges / descriptions).
    The surfaces still write their own constraints, and parity tests assert they agree with the
    spec (guard, not derive); the job runner's start check, the Training page's base form and the
    Fine-tune page check values with its `value_problem`. The record above is left as written.
  - *Note (2026-11-12, ops-wiring round A1a):* what a new run takes is stated once
    (`src/utils/run_plan.py` `new_run_settings`, the training process's rule) and judged by one
    check (`new_run_problem`: the settings a run does not read, each value by `value_problem`,
    the rules across settings), which the training process and the job runner's starts make
    alike. The API's `TrainingStartRequest` is a thin projection of it: it offers every setting
    a new base reads, checks each value first by `value_problem` (strictly, in its words) and
    **derives its bounds from the field spec** (`FieldSpec.valid`, for the schema); the parity
    tests still compare the two. The rules across settings are data in the spec
    (`CROSS_FIELD_RULES`) with one evaluator (`cross_field_problems`), which that check and a
    branch's plan (on what it changes) call; the Trainer refuses the onecycle rule in the same
    sentence. The WebUI's base form and Fine-tune page still offer part of the settings and
    check values with `value_problem`; the rules across settings are judged at the start (the
    pages follow in B1 and B2).
  - *Note (2026-11-12, checkpoint A's verification):* `TrainingStartRequest` now carries every
    learning setting the field spec defines (`run_plan.RUN_SETTINGS`) and the task, each
    optional with the field spec's bounds: a new base still refuses what a base does not read
    in `new_run_problem`'s sentence, and a continuation's stated fields go to the runner, which
    judges them as the training process does (`run_plan.plan_continuation`: the parent's stage
    and task by `parent_identity_problem`, then the settings stated by `stated_problem`, then
    the parent's settings). The runner resolves the training device by the process's resolver
    (`src/utils/training_device.py`) before any start.
  - *Note (2026-11-13, ops-wiring round B1a):* the Training page's base form offers what a new
    base takes (`new_run_settings("base")`), each field starting at the one fill's value
    (`new_run_values`), and the reference spec's input, encoder and LP head fields generated
    from their dataclasses; a setting a rule across settings reads only under its guard shows
    only while it applies (`training_fields.not_read_by_rules`, the rules' own guard). The base
    form and the Fine-tune page send a setting when the operator touched its field, whatever its
    value (decision S2); the K6 rule of sending only what differs from the reference retired.
    The help leads are the field spec's `description`.

## Rationale

Centralization ≠ modularity. The project principle is **"one clear home per concern,"** not
"all config in one global file." Training, inference, and model configs are *different concerns*
and belong with their owning modules. A single global manager would re-couple those layers
(training ↔ inference ↔ models ↔ API ↔ UI) — precisely the coupling this project avoids.

The planned *architecture* (layered, protocol-first, one-home-per-concern) was sound; a single
implementation attempt (`hyperparameters.py`) that was never wired in is not the same as that
architecture, and keeping it while the dataclasses do the real work created two homes for one
concern — a modularity violation. Removing it **serves** the principle.

## Consequences / related dispositions

- **`core/types.py` `ModelConfig` / `TrainingConfig` / `DataConfig`:** legacy *protocol-surface*
  types (referenced by `ConfigLoaderProtocol` and `TrainerProtocol` in `core/protocols.py`, and
  re-exported by `core/__init__.py`) — **not** runtime authorities. Cleanup **deferred** to the
  future protocols refactor (avoid churning the `protocols` hub for limited immediate value).
  *(2026-11-09: removed in K7, 2026-10-29; `ConfigLoaderProtocol` and `TrainerProtocol`, which
  named them, remain in `core/protocols.py`.)*
- **`runtime_presets.py`:** real runtime-settings source of truth (allocator presets / launch
  settings); **relocated to `src/config/runtime_presets.py`** (PR-2) — `REPO_ROOT` bumped to
  `parents[2]`; all importers/tests updated.
- **`config_validator.py` / `schema_loader.py`:** kept as **documented RESERVED homes** for
  committed-config validation (`configs/deployment.yaml` + `configs/schemas/*.json` exist but are
  not yet validated). Empty by design until deployment-config validation is built.

## Rollout (behavior-preserving; each step an independent, revertible PR)

| PR | Change |
|---|---|
| PR-1 ✅ | Remove `hyperparameters.py` (+ update `config/__init__.py`, `run_local_tests.py`, this record) |
| PR-2 ✅ | Move `runtime_presets.py` → `src/config/runtime_presets.py` |
| PR-3 ✅ | Docstring the reserved `config_validator.py` / `schema_loader.py` |
| PR-4 | B-lite shared training field-spec (with API/WebUI/`TrainerConfig` parity tests) |

*(2026-11-09: PR-4 is partly done, as 4a–4c (`5b1c846` to `dd999e3`): the surfaces are guarded by
the parity tests, not derived from the spec; the note under the decision above.)*
*(2026-11-12: the API's start derives its bounds from the spec since the ops-wiring round's
A1a; the pages' fields follow in its stages B1 and B2; the note under the decision above.)*
