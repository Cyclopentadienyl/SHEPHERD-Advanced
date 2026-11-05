# SHEPHERD-Advanced Directory Structure Blueprint

**Last Updated**: 2026-04-28
**Status**: Canonical Reference

This document defines the official directory layout for the SHEPHERD-Advanced project.

---

## Top-Level Layout

```
SHEPHERD-Advanced/
├── configs/                # Configuration files (YAML, JSON)
├── data/                   # All data: external downloads + workspaces
├── docs/                   # Documentation
├── logs/                   # Training logs & real-time progress
├── models/                 # Only KG-independent pretrained models
├── outputs/                # Training outputs (final metrics, results)
├── scripts/                # Entry-point scripts (train, evaluate, infer)
├── src/                    # Source code (7-layer architecture)
└── tests/                  # Test suite (unit, integration, e2e)
```

---

## Core Concept: Workspaces

A **workspace** is a self-contained directory holding one KG version and all
training artifacts derived from it. Everything a model needs for inference
lives in the same folder.

**Key rule**: one KG version = one workspace. Checkpoints trained on a
different KG version belong in a different workspace.

```
data/workspaces/{name}/
├── kg.json                  # Knowledge graph structure
├── node_features.pt         # Node feature vectors (GNN input)
├── edge_indices.pt          # Edge connectivity (with reverse edges)
├── num_nodes.json           # Node counts per type
├── runs/                    # Training runs on THIS KG, one folder per run (M3a-2)
│   └── {conv_type}/
│       └── {date-time}_{run name}_{run id}/
│           ├── config.yaml          # the settings in effect
│           ├── runtime.json         # the allocator
│           ├── final_metrics.json   # written when the run ends (not when stopped)
│           ├── checkpoints/         # last.pt (every epoch), model-<epoch>-<metric>.pt
│           └── logs/                # train_*.json, progress.json
└── archive/                 # Old checkpoints contract M2 refuses, where an
                             # earlier move put them (kept as written)
```

No run writes into another run's folder, a continued run included: it gets a
new folder and records its parent by SHA-256.

### Workspace Lifecycle

```
1. Data Preparation (run once per KG version):
   build_knowledge_graph.py → kg.json
   kg.export_graph_data()   → node_features.pt, edge_indices.pt, num_nodes.json

2. Training (may run many times with different hyperparameters):
   train_model.py reads from workspace root, writes to its own runs/{conv}/{run}/

3. Inference (ongoing):
   Pipeline loads kg.json + graph data + chosen checkpoint
   All from the same workspace directory
```

### Relationship Between Files

| File | When Created | Changes During Training? | Purpose |
|------|-------------|------------------------|---------|
| `kg.json` | Data prep (once) | No | KG structure (nodes + edges) |
| `node_features.pt` | Data prep (once) | No | GNN input features |
| `edge_indices.pt` | Data prep (once) | No | Graph connectivity |
| `num_nodes.json` | Data prep (once) | No | Node count statistics |
| `runs/{conv}/{run}/checkpoints/*.pt` | Every training epoch | **Yes** (new runs) | Learned GNN weights |

The first 4 files are **static** — shared by all training runs within the
workspace. Only run folders accumulate over time.

An older workspace may still hold `shortest_paths.pt` and
`shortest_paths.meta.json`; this build leaves the pair in place and reads
nothing from it ([deployment guide](../deployment-guide.en.md) §4.3).

### KG Version Compatibility

Each checkpoint embeds a **data fingerprint** (node types, counts, edge types),
a structural summary of the graph the run read; nothing compares it at load
(since K8-4). What binds a checkpoint to its workspace is its training record
(`training_input_digests`): unless it names this workspace's graph files by
digest, the load is refused before any model is built (contract M2.4; the
[playbook](TRAINING_PIPELINE_PLAYBOOK.md)'s safety net).

---

## Directory Details

### `data/` — All Data

```
data/
├── external/                # Large external datasets (git-ignored)
│   ├── README.md            # Download instructions
│   └── .gitignore
└── workspaces/              # Workspace directories
    ├── .gitkeep
    ├── demo/                # Development/testing workspace
    └── (future workspaces)
```

### `models/` — KG-Independent Models Only

```
models/
└── pretrained/              # Pre-trained base models (e.g., ClinicalBERT)
    └── README.md
```

KG-specific checkpoints live inside their workspace's `runs/` directory, NOT in
`models/`.

### `configs/` — Configuration Files

```
configs/
├── deployment.yaml          # Unified deployment configuration
├── accelerators.json        # GPU accelerator specifications
└── schemas/                 # JSON schemas for validation
```

### `logs/` — Training Logs

```
logs/
├── progress.json            # Real-time batch-level progress
└── train_YYYYMMDD_HHMMSS.json  # Epoch-level metrics history
```

### `outputs/` — Training Outputs

```
outputs/
└── training_metrics_YYYYMMDD.csv  # Exported metrics CSV
```

---

## Path Configuration

### In deployment.yaml

```yaml
paths:
  workspaces_root: data/workspaces/
  default_workspace: data/workspaces/default
  pretrained_root: models/pretrained/
  logs_root: logs/
  cache_root: .cache/
```

### In WebUI

The Diagnosis tab's **Model Configuration** accordion allows selecting a
workspace directory. The pipeline auto-detects `kg.json`, graph data files,
and checkpoints within that workspace.

### In Training Console

The **Data Directory** field points to a workspace. Each run gets a new folder
under `{workspace}/runs/{conv_type}/`, named after the optional **Run name**;
Status shows it once training starts. **Continue Training** lists the
architecture's runs, or takes a typed checkpoint path, through one check.

---

## Git Tracking Strategy

| Directory | Tracked | Strategy |
|---|---|---|
| `configs/` | Yes | All config files tracked |
| `data/external/` | Structure only | README tracked, data files ignored |
| `data/workspaces/` | Structure only | `.gitkeep` tracked, all data ignored |
| `models/pretrained/` | README only | Model files too large for git |
| `logs/` | Structure only | Log files ignored |
| `outputs/` | Structure only | Artifacts ignored |

---

## Migration from Legacy Layout

If upgrading from the old `data/processed/` + `models/checkpoints/` layout:

```
# Move legacy data into a workspace
mkdir data/workspaces/legacy
mv data/processed/*.pt data/workspaces/legacy/
mv data/processed/*.json data/workspaces/legacy/
mv models/checkpoints/*.pt data/workspaces/legacy/checkpoints/
```

Checkpoints in a workspace's older `checkpoints/` folder are not selected
automatically (no runtime fallback, decision 9); they stay on disk, reachable by
explicit path. One without a model spec (every file written before the model
bundles) is refused at load, before any model is built.
