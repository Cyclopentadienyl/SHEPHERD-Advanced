"""
A synthetic data directory and checkpoint, of a shape no sampling can change.
======================================================================
The graph is shaped so that no draw could matter:

  - **every node's degree is at or below the neighbour-sampling limit**
    (`num_neighbors = [15, 10, 5]`), so a sampler never has a choice to make;
  - **every gene links every phenotype to every disease**, so the 2-hop expansion
    from any disease seed reaches all diseases.

It was shaped for the ShepherdGNN path's sampling loader, which seeded nothing;
that loader and the tests that drove it retired in K8-7, and nothing that reads
this workspace samples now. A task measurement scores every disease, so the
shape matters to it only in that it is degenerate: no rank from this workspace
is evidence about the scorer.

The checkpoint is `write_checkpoint_without_model_spec` over this workspace
(`tests/fixtures/registered_cohort.py`), written without any model class: no
weights and no model spec, only its `epoch`, its `logs` and the
`training_input_digests` record a training run over the workspace writes. Every
build over it is refused before a model is built, so the tests that use it read
its data, its bytes or its refusal, never a model (K8-10, F2).

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
#: Cases per disease in each generated split: two, so `val` holds four cases.
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
    """Write a data directory and a checkpoint without a model spec. Returns
    ``(data_dir, checkpoint)``.

    The layout is the one the shared readers read (`read_graph_artifacts`,
    `read_samples`), with the manifest that binds the graph. The checkpoint is
    the one the module docstring describes.

    **The size arguments default to the original fixture**, so every existing
    caller gets exactly the workspace it got before. They exist because a
    four-disease cohort cannot exercise anything that only happens above twenty
    candidates: a top-20 window and no window are the same list at four columns.
    See `_graph` for the precondition a wider workspace carries.
    """
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
    from tests.fixtures.generated_workspace import write_generated_workspace

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

    from tests.fixtures.registered_cohort import write_checkpoint_without_model_spec

    checkpoint_path = root / "checkpoint.pt"
    write_checkpoint_without_model_spec(checkpoint_path, data_dir=data_dir)
    return data_dir, checkpoint_path
