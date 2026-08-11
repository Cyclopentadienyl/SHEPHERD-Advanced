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

Three steps, in order. Full walkthrough with data sources and expected artifacts:
[`docs/TRAINING_PIPELINE_PLAYBOOK.md`](docs/TRAINING_PIPELINE_PLAYBOOK.md).

| Step | Command | Produces |
|---|---|---|
| 1. Knowledge graph | `scripts/build_knowledge_graph.py` | `kg.json`, `node_features.pt`, `edge_indices.pt` |
| 2. Shortest paths | `scripts/compute_shortest_paths.py` | `shortest_paths.pt` |
| 3. Train | dashboard training console, or `scripts/train_model.py` | `checkpoints/*.pt` |

Artifacts live under `data/workspaces/<kg>/`, keeping each knowledge graph's outputs separate from
the KG-independent models in `models/pretrained/`. See
[`docs/DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md).

> **There is no step 4.** Building a vector index used to be listed here. The vector-index
> subsystem has been detached from diagnosis — it is kept, implemented and tested, for planned
> natural-language and vector-mapping work, and is built with `make vector-index` when that work
> needs it. It is not part of building a model. See
> [`docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md).

## Development

The development tooling — pytest, ruff, mypy, import-linter — is in the `dev` extra, which
`deploy.sh` does **not** install: it syncs runtime dependencies only. On a machine that has only
been deployed, install them first, or `make check` stops at `lint-imports` with `import-linter is
not installed`:

```bash
uv sync --inexact --extra dev
```

`--inexact` matters: plain `uv sync` removes packages that are not in `uv.lock`, and `deploy.sh`
installs the PyG native extensions and `cuvs-cu13` outside it. `deploy.sh` syncs with `--inexact`
for the same reason.

```bash
make check          # the gate: lint-imports + test-unit
make test           # full suite      make test-unit / make test-integration
make lint-imports   # layered-architecture check
```

Run `make help` for every target. Optional local hooks: `uv run pre-commit install`.

**`make check` is the gate** and is expected to pass. It contains only checks that are green on the
current tree, so a red result means something you changed.

`make lint-imports` enforces the layered architecture mechanically (`.import-linter.ini`): a lower
layer may never import a higher one, and the WebUI may not import the training stack directly. These
rules were documented as "enforced" long before anything ran them; they are now actually checked.

**Known-red commands.** These are useful for measuring debt, but they do **not** pass today and are
deliberately excluded from `make check`:

| Command | Current state |
|---|---|
| `make lint` (ruff) | large pre-existing backlog; most findings are auto-fixable via `make format`, but that rewrites annotations across `src/` and needs its own reviewed change |
| `make typecheck` (mypy `--strict`) | large pre-existing backlog; the codebase predates any type gate |
| `make test-integration` | passes on the development container, but has not been demonstrated green across all supported platforms; promoting it into `check` is a separate, measured decision |

Clearing these backlogs is tracked as separate work. They were not silenced with blanket ignores:
a check that passes without checking anything is worse than one that honestly reports debt.

Tests are invoked as `python -m pytest` so the repository root stays on `sys.path` and
`import src...` resolves without an editable install. Tests that need optional dependencies
(torch, gradio, fastapi, cuVS) skip cleanly when those are absent.

## Documentation

Start here for the living documents:

| Document | Contents |
|---|---|
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layered design, scoring model, design principles |
| [`DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md) | Where artifacts live and why |
| [`TRAINING_PIPELINE_PLAYBOOK.md`](docs/TRAINING_PIPELINE_PLAYBOOK.md) | End-to-end build walkthrough |
| [`CONFIG_AUTHORITY.md`](docs/CONFIG_AUTHORITY.md) | Which module owns which configuration, and why |
| [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) | Open architecture findings under review |
| [`GNN_ARCHITECTURE_NOTES.md`](docs/GNN_ARCHITECTURE_NOTES.md) | Model design notes |
| [`module_dependencies.md`](docs/module_dependencies.md) | Inter-module dependency map |

[`docs/README.md`](docs/README.md) indexes **every** Markdown document in the repository and labels
each one **Living**, **Dated snapshot**, or **Archived** — several documents under `docs/` are
point-in-time reports that are deliberately not kept current, and the index is how you tell which
is which. It also lists the validation and diagnostic scripts that nothing runs automatically.

Project-level planning documents (`medical-kg-blueprint.md`, `medical-kg-todo.md`,
`deployment-guide.md`) stay at the repository root and are referenced from several documents by
path.

## Licence

**No open-source licence is granted.** The repository is public for transparency and review, not
for redistribution — see [`LICENSE`](LICENSE).

This is a decision, not a gap. The project is built for one deploying institution and tailored to
that deployment, so attaching an open-source licence to it before anyone has asked to use it would
grant rights nobody requested. If you want to use or collaborate on this project, open an issue —
the licence will be chosen then.

`pyproject.toml` previously declared MIT while `LICENSE` was empty, announcing a grant that had
never been made. It now declares no licence at all.
