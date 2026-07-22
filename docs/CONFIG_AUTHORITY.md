# Config Authority — Decision Record

**Status:** Accepted (2026-07). Supersedes the unused `HyperparameterManager`.

## Decision — Model B + B-lite

**Runtime configuration authority stays with the per-concern dataclasses:**

| Concern | Authority | Location |
|---|---|---|
| Training runtime | `TrainerConfig` | `src/training/trainer.py` |
| Inference runtime | `PipelineConfig` | `src/inference/pipeline.py` |
| Model architecture | `ShepherdGNNConfig` | `src/models/gnn/shepherd_gnn.py` |

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
| PR-3 | Docstring the reserved `config_validator.py` / `schema_loader.py` |
| PR-4 | B-lite shared training field-spec (with API/WebUI/`TrainerConfig` parity tests) |
