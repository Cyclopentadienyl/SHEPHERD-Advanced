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
- **Hardware acceleration:** required to train and to serve a model; CUDA (NVIDIA) is currently the
  only supported backend. An unset or `auto` device means CUDA and is refused on a host without it; `cpu` (`--device cpu` for training, `SHEPHERD_DEVICE=cpu` for
  serving) is an explicit choice for development, recorded as such (deployment guide §4.8).

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

Both start the FastAPI service with the Gradio dashboard mounted on it. Neither reads an
environment file: give the service its settings in the shell that runs the launcher, or in the
systemd unit, as [deployment guide §4.12](deployment-guide.en.md) states them. Without them the
service starts read-only and lists nothing to train or test: the operator controls (Start, Load,
a test run) need `SHEPHERD_DEPLOYMENT_MODE=M1`, every training and test run needs
`SHEPHERD_WORKSPACES_ROOT`, the GPU lock's one name (§4.11), and the Test page needs
`SHEPHERD_REGISTRY`, the deployment's registry (and `SHEPHERD_INTAKE_DIR` to import a cohort).

| | |
|---|---|
| Dashboard (Diagnosis; Research › Datasets, Training, Test, Models; Settings) | http://127.0.0.1:8000/ui |
| API docs (Swagger) | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

## Build a model

Three steps, in order. Full walkthrough with data sources and expected artifacts:
[`docs/TRAINING_PIPELINE_PLAYBOOK.md`](docs/TRAINING_PIPELINE_PLAYBOOK.md).

| Step | Command | Produces |
|---|---|---|
| 1. Knowledge graph | `scripts/build_knowledge_graph.py` | `kg.json`, `node_features.pt`, `edge_indices.pt` |
| 2. Shortest paths | `scripts/compute_shortest_paths.py` | `shortest_paths.pt` |
| 3. Train | the dashboard's Research › Prepare › Training, or `scripts/train_model.py --data-dir <workspace>` in a shell given the service's `SHEPHERD_WORKSPACES_ROOT` (deployment guide §4.11) | `runs/<arch>/<run>/checkpoints/*.pt` |

Artifacts live under `data/workspaces/<kg>/`, keeping each knowledge graph's outputs separate from
the KG-independent models in `models/pretrained/`. See
[`docs/DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md).

> **There is no step 4.** Building a vector index used to be listed here. The vector-index
> subsystem has been detached from diagnosis — it is kept, implemented and tested, for planned
> natural-language and vector-mapping work, and is built with `make vector-index` when that work
> needs it. It is not part of building a model. See
> [`docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md).

## Test a model

A checkpoint is tested on a registered test cohort version, from the CLI below or from the
dashboard's Research › Evaluate › Test, and its result is recorded beside it (provenance contract
M4; deployment guide §4.9). Run the commands in a shell given the service's settings (deployment
guide §4.11): `SHEPHERD_REGISTRY` naming the deployment's registry, the `<registry>` below
(registration and `--cohort` refuse any other, and refuse while it is unset), and, for step 2,
`SHEPHERD_WORKSPACES_ROOT`, the GPU lock's one name.

| Step | Command | Produces |
|---|---|---|
| 1. Import a cohort | `scripts/import_cohort.py --registry <registry> --data-dir <workspace> --source <cases.json> --label <label> --role <role>`: a preview, then the same with `--commit <preview id>` | an immutable cohort version, mapped to one workspace's graph, in a registry outside the repository |
| 2. Measure | `scripts/measure_scorer.py --checkpoint <file> --data-dir <workspace> --cohort <label> --registry <registry>` | a Mode C report, kept in the registry's report store |
| 3. Record | `--record` on step 2, or later `scripts/record_evaluation.py --report <report> --checkpoint <file>`, the command step 2 prints | a record in `evaluations.json` beside the checkpoint's resolved path, where the service reads it |

The Diagnosis page's "Model in use" card shows the loaded checkpoint's recorded results as a
state (how many, how many contested); Research › Evaluate › Models lists them, and API clients
read them through `GET /api/v1/evaluations/loaded-model`. They are the model's own Mode C disease
ranking: until work item B-1 ships, the Diagnosis page itself ranks by the GNN + shortest-path
mixture above, so a recorded number does not describe its ranking.

## Development

The development tooling — pytest, ruff, mypy, import-linter — is in the `dev` extra, which
`deploy.sh` does **not** install: it syncs runtime dependencies only. On a machine that has only
been deployed, install them first, or `make check` stops at `lint-imports` with `import-linter is
not installed`:

```bash
uv sync --inexact --extra dev
```

`--inexact` matters: it stops uv removing the out-of-lock PyG native extensions and `cuvs-cu13`
simply for being extraneous, which is how `deploy.sh` installs them. (It is not an unconditional
guarantee — a package that conflicts with a project dependency can still be replaced.) `deploy.sh`
and `deploy.cmd` sync with `--inexact` for the same reason.

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
| [`DISEASE_SCORER_POLICY.md`](docs/DISEASE_SCORER_POLICY.md) | Decision record: what ranks disease candidates, and the role of the shortest-path signal |
| [`SP_SCORE_GUIDE.md`](docs/SP_SCORE_GUIDE.md) | **For clinicians:** what the shortest-path score means, and when it should and should not be consulted |
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
