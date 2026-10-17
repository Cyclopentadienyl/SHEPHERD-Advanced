"""
A synthetic data directory and checkpoint, built so sampling randomness cannot
change the answer.
======================================================================
The dataloader seeds nothing: neighbour sampling uses `random.sample` /
`random.choice` (`src/kg/data_loader.py:220,263`) and negatives use
`np.random.choice` / `random.randint` (`:534-597`). Two processes running the
same command can therefore build different subgraphs.

Rather than seed the RNG — which would mean reaching into production code — the
graph is shaped so the draw cannot matter:

  - **every node's degree is at or below the neighbour-sampling limit**
    (`num_neighbors = [15, 10, 5]`), so `random.sample` never has a choice to
    make;
  - **every gene links every phenotype to every disease**, so the 2-hop expansion
    from any disease seed reaches all diseases. Whichever negatives are drawn,
    the candidate universe is the same set.

The tests that drive the sampling loader over this workspace (the Trainer's
validation values, `tests/unit/test_trainer_validation_values.py`) rely on that
shape. Mode C uses no sampler: it scores every disease, so the shape matters to
it only in that it is degenerate — no rank from this workspace is evidence about
the scorer.

The checkpoint is written in the shape the training writer produces (weights,
config, epoch, logs, and the graph roles of `training_input_digests`): there is
no `metadata` or `in_channels_dict` key, so every test built on it builds the
model the way production does.

Module: tests/fixtures/synthetic_workspace.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Small enough that every degree sits under the sampling limits, and complete
# enough that the 2-hop disease universe is the whole disease set.
N_PHENOTYPES = 3
N_GENES = 2
N_DISEASES = 4
N_SAMPLES = 6
FEATURE_DIM = 8
HIDDEN_DIM = 8


def _graph(
    n_phenotypes: int = N_PHENOTYPES, n_diseases: int = N_DISEASES
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], Dict[str, int]]:
    """The complete phenotype-gene-disease tripartite graph, at any two sizes.

    **The sizes are parameters, and the stability precondition is the caller's to
    keep.** A gene's out-degree here is `n_diseases`, so a workspace wider than the
    consumer's `num_neighbors` limit turns the 2-hop expansion into a real draw and
    the candidate universe stops being seed-independent. The defaults sit under the
    library default of `[15, 10, 5]`; a caller asking for more must widen
    `num_neighbors` to match, as the wide Trainer cohort does.
    """
    torch.manual_seed(20260818)
    x_dict = {
        "phenotype": torch.randn(n_phenotypes, FEATURE_DIM),
        "gene": torch.randn(N_GENES, FEATURE_DIM),
        "disease": torch.randn(n_diseases, FEATURE_DIM),
    }

    ph_gene = torch.tensor(
        [[p, g] for p in range(n_phenotypes) for g in range(N_GENES)], dtype=torch.long
    ).t()
    gene_dis = torch.tensor(
        [[g, d] for g in range(N_GENES) for d in range(n_diseases)], dtype=torch.long
    ).t()

    edge_index_dict = {
        ("phenotype", "associated_with", "gene"): ph_gene,
        ("gene", "causes", "disease"): gene_dis,
    }
    num_nodes_dict = {
        "phenotype": n_phenotypes,
        "gene": N_GENES,
        "disease": n_diseases,
    }
    return x_dict, edge_index_dict, num_nodes_dict


def _samples(
    n_phenotypes: int = N_PHENOTYPES,
    n_diseases: int = N_DISEASES,
    n_samples: int = N_SAMPLES,
) -> List[dict]:
    """Deterministic and hand-written — no RNG, so sample order is fixed."""
    return [
        {"patient_id": f"P{i:02d}",
         "phenotype_ids": [i % n_phenotypes, (i + 1) % n_phenotypes],
         "disease_id": i % n_diseases}
        for i in range(n_samples)
    ]


#: The generated cut: two diseases for training, the next two for validation.
#: Disjoint, as the allocation step guarantees, and inside the smallest graph
#: this fixture builds.
GENERATED_TRAIN_DISEASES = (0, 1)
GENERATED_VAL_DISEASES = (2, 3)
#: Cases per disease in each generated split: enough that a Mode C run over
#: `val` (four cases) spans more than one chunk at the tests' batch size of 3.
GENERATED_CASES_PER_DISEASE = 2


def _generated_diseases(n_diseases: int) -> Tuple[List[int], List[int]]:
    if n_diseases <= max(GENERATED_VAL_DISEASES):
        raise ValueError(
            f"the generated cut needs at least {max(GENERATED_VAL_DISEASES) + 1} diseases, "
            f"got {n_diseases}"
        )
    return list(GENERATED_TRAIN_DISEASES), list(GENERATED_VAL_DISEASES)


def _generated_samples(split: str, disease_ids: List[int], n_phenotypes: int) -> List[dict]:
    """Rows whose ids are nodes of this graph, two per disease, deterministic.
    The patient ids carry `SECRET` like the generated fixture's, so a leak of one
    into an artifact is findable."""
    return [
        {"patient_id": f"SECRET-{split}-{d}-{j}",
         "phenotype_ids": [(d + j) % n_phenotypes, (d + j + 1) % n_phenotypes],
         "disease_id": d}
        for d in disease_ids for j in range(GENERATED_CASES_PER_DISEASE)
    ]


def build_workspace(
    root: Path,
    split: str = "test",
    n_phenotypes: int = N_PHENOTYPES,
    n_diseases: int = N_DISEASES,
    n_samples: int = N_SAMPLES,
) -> Tuple[Path, Path]:
    """Write a data directory and a checkpoint. Returns ``(data_dir, checkpoint)``.

    The layout is the one the shared readers read (`read_graph_artifacts`,
    `read_samples`), with the manifest that binds the graph.

    **The size arguments default to the original fixture**, so every existing
    caller gets exactly the workspace it got before. They exist because a
    four-disease cohort cannot exercise anything that only happens above twenty
    candidates — the Trainer keeps the first twenty columns of its validation
    ranking, and a top-20 window and no window are the same list at four columns.
    See `_graph` for the precondition a wider workspace carries.
    """
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    x_dict, edge_index_dict, num_nodes_dict = _graph(n_phenotypes, n_diseases)
    torch.save(x_dict, data_dir / "node_features.pt")
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")
    (data_dir / "num_nodes.json").write_text(json.dumps(num_nodes_dict))
    (data_dir / f"{split}_samples.json").write_text(
        json.dumps(_samples(n_phenotypes, n_diseases, n_samples))
    )

    # **The graph is bound by a real manifest, and the generated cohorts fit it.**
    # The `test` file above is a plain samples file the readers and the Trainer's
    # tests read directly; it is no cohort a measurement resolves (a supplied
    # cohort is a registered version since M4, which needs a `kg.json` it can map
    # against, `tests/fixtures/registered_cohort.py`). The generated `train` and
    # `val` beside it are cut by `write_generated_workspace`, bound by the
    # manifest, and hold ids inside this graph, so `--split val` measures them.
    from tests.fixtures.generated_workspace import graph_record, write_generated_workspace

    (data_dir / "kg.json").write_text(json.dumps({"synthetic": True}))
    train_ids, val_ids = _generated_diseases(n_diseases)
    write_generated_workspace(
        data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        profiles={d: {"phenotype_ids": [d % n_phenotypes], "gene_ids": [d % N_GENES]}
                  for d in train_ids + val_ids},
        train_samples=_generated_samples("train", train_ids, n_phenotypes),
        val_samples=_generated_samples("val", val_ids, n_phenotypes),
    )

    metadata = (list(num_nodes_dict), list(edge_index_dict))
    in_channels_dict = {k: FEATURE_DIM for k in num_nodes_dict}
    config = ShepherdGNNConfig(hidden_dim=HIDDEN_DIM, num_layers=2, num_heads=2)

    torch.manual_seed(20260818)
    model = ShepherdGNN(metadata=metadata, in_channels_dict=in_channels_dict, config=config)

    checkpoint_path = root / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"hidden_dim": HIDDEN_DIM, "num_layers": 2, "num_heads": 2},
            "epoch": 1,
            "logs": {"val_mrr": 0.0},
            # The graph roles training records (M2.2), so serving and
            # measurement accept it for this workspace (M2.4).
            "training_input_digests": graph_record(data_dir),
        },
        checkpoint_path,
    )
    return data_dir, checkpoint_path
