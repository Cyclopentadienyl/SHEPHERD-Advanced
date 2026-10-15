"""A workspace `scripts/train_model.py` can actually train.

The synthetic fixtures elsewhere cannot complete a training: their generated
profiles name phenotypes `d` and `d + 100`, outside the three-phenotype graph
they are paired with, and data loading stops at an IndexError. This one builds
the same complete tripartite graph with every sample's ids inside it. Moved here
from `tests/integration/test_train_model_run.py` so the unit tests that train
for real (contract M3a's checkpoint record) use the same workspace.

Module: tests/fixtures/trainable_workspace.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def write_trainable_workspace(root: Path) -> Path:
    from tests.fixtures.generated_workspace import write_generated_workspace
    from tests.fixtures.synthetic_workspace import N_GENES, _graph

    n_phenotypes, n_diseases = 6, 6
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    x_dict, edge_index_dict, num_nodes = _graph(n_phenotypes, n_diseases)
    torch.save(x_dict, data_dir / "node_features.pt")
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")
    (data_dir / "num_nodes.json").write_text(json.dumps(num_nodes))
    (data_dir / "kg.json").write_text(json.dumps({"synthetic": True}))

    profiles = {
        d: {"phenotype_ids": [d % n_phenotypes, (d + 1) % n_phenotypes],
            "gene_ids": [d % N_GENES]}
        for d in range(n_diseases)
    }

    def samples(split, ids, per_disease):
        # Both optional fields training reads (`training_fields=True`).
        return [
            {"patient_id": f"{split}-{d}-{i}",
             "phenotype_ids": list(profiles[d]["phenotype_ids"]),
             "disease_id": d,
             "gene_ids": list(profiles[d]["gene_ids"]),
             "candidate_disease_ids": sorted(ids)}
            for d in ids for i in range(per_disease)
        ]

    train_ids, val_ids = [0, 1, 2, 3], [4, 5]
    write_generated_workspace(
        data_dir, train_ids=train_ids, val_ids=val_ids, profiles=profiles,
        train_samples=samples("train", train_ids, 3),
        val_samples=samples("val", val_ids, 2),
    )
    return data_dir
