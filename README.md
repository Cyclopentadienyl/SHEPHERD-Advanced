# SHEPHERD-Advanced

Phenotype-driven diagnostic reasoning for rare disease. A heterogeneous graph neural network is
trained over a medical knowledge graph (HPO phenotypes, MONDO diseases, genes, pathways,
cross-species orthologs) and ranks candidate diseases from a patient's HPO terms. A path reasoner
supplies the evidence chains a clinician can inspect, so a ranking is never presented without an
explanation.

> **Status:** research/engineering build. Not a certified medical device and not for autonomous
> clinical decision-making. Output is decision *support* for a qualified clinician.

## How it works

```
patient HPO terms
   │
   ├─► GNN embeddings ──────► similarity score  ┐
   │                                            ├─► η·embedding + (1−η)·shortest-path → ranking
   ├─► shortest-path table ─► proximity score   ┘
   │
   └─► PathReasoner ───────► evidence paths (phenotype → gene → disease) + confidence label
```

Scoring is GNN-primary (η defaults to 0.7); the shortest-path signal and the path evidence are
complementary. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the layered design and the
design principles it must satisfy.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **Platforms:** Windows x86-64, Linux x86-64, Linux aarch64 (NVIDIA DGX Spark)
- **GPU:** optional. CUDA accelerates training; inference runs on CPU.

## Install

```bash
# Linux (x86 / ARM)
bash deploy.sh

# Windows
deploy.cmd
```

`deploy.sh` resolves the PyTorch/PyG stack for the detected platform and CUDA line, then runs
installation validation. Verify an existing environment at any time:

```bash
make validate          # or: uv run python scripts/validate_installation.py
```

## Run

```bash
bash launch_shepherd.sh          # Linux
launch_shepherd.cmd              # Windows
```

Both start the FastAPI service with the Gradio dashboard mounted on it:

| | |
|---|---|
| Dashboard (training console, diagnosis panel, runtime settings) | http://127.0.0.1:8000/ui |
| API docs (Swagger) | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

## Build a model

Four steps, in order. Full walkthrough with data sources and expected artifacts:
[`docs/TRAINING_PIPELINE_PLAYBOOK.md`](docs/TRAINING_PIPELINE_PLAYBOOK.md).

| Step | Command | Produces |
|---|---|---|
| 1. Knowledge graph | `scripts/build_knowledge_graph.py` | `kg.json`, `node_features.pt`, `edge_indices.pt` |
| 2. Shortest paths | `scripts/compute_shortest_paths.py` | `shortest_paths.pt` |
| 3. Train | dashboard training console, or `scripts/train_model.py` | `checkpoints/*.pt` |
| 4. Vector index *(optional)* | `scripts/build_index.py` | ANN index — see the caveat below |

Artifacts live under `data/workspaces/<kg>/`, keeping each knowledge graph's outputs separate from
the KG-independent models in `models/pretrained/`. See
[`docs/DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md).

> **Step 4 caveat.** The vector-index subsystem is under review: it has never been active in the
> audited environment and has known defects. Do not rely on it. See
> [`docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md).

## Development

```bash
make test           # full test suite     make lint          # ruff
make test-unit      # unit tests only     make format        # black + ruff --fix
make typecheck      # mypy                make lint-imports  # layered-architecture check
make check          # lint + lint-imports + typecheck + test
```

Run `make help` for every target. Optional local hooks: `uv run pre-commit install`.

`make lint-imports` enforces the layered architecture mechanically (`.import-linter.ini`): a lower
layer may never import a higher one, and the WebUI may not import the training stack directly. The
rules were previously documented as "enforced" but nothing ran them — they are now checked.

Tests are invoked as `python -m pytest` so the repository root stays on `sys.path` and
`import src...` resolves without an editable install. Tests that need optional dependencies
(torch, gradio, fastapi, cuVS) skip cleanly when those are absent.

## Documentation

| Document | Contents |
|---|---|
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layered design, scoring model, design principles |
| [`DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md) | Where artifacts live and why |
| [`TRAINING_PIPELINE_PLAYBOOK.md`](docs/TRAINING_PIPELINE_PLAYBOOK.md) | End-to-end build walkthrough |
| [`CONFIG_AUTHORITY.md`](docs/CONFIG_AUTHORITY.md) | Which module owns which configuration, and why |
| [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) | Open architecture findings under review |
| [`GNN_ARCHITECTURE_NOTES.md`](docs/GNN_ARCHITECTURE_NOTES.md) | Model design notes |
| [`module_dependencies.md`](docs/module_dependencies.md) | Inter-module dependency map |

Project-level planning documents (`medical-kg-blueprint.md`, `medical-kg-todo.md`,
`deployment-guide.md`) stay at the repository root and are referenced from several documents by
path.

## Licence

`pyproject.toml` declares MIT, but the `LICENSE` file is currently empty — the licence is **still
being confirmed** with the deploying institution. Treat the licensing status as unresolved until
that file is populated.
