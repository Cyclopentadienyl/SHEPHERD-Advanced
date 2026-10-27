"""A workspace the base stage (KG link prediction) can train on.

`trainable_workspace` has two relations without reverse twins and two
dozen edges, too few for a validation split in which every relation has a
second tail. This graph is the shape the KG builder exports: each relation
with its `rev_` twin, three node types, a few hundred edges, one component.
Samples are written only because a workspace's manifest binds them; the base
stage does not read them.

Module: tests/fixtures/base_workspace.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def write_base_workspace(root: Path, *, seed: int = 7) -> Path:
    from tests.fixtures.generated_workspace import write_generated_workspace

    rng = random.Random(seed)
    n_ph, n_gene, n_dis = 24, 16, 8
    data_dir = root / "data"
    data_dir.mkdir(parents=True)

    def edges(n_src, n_dst, per_src):
        pairs = set()
        for s in range(n_src):
            for d in rng.sample(range(n_dst), per_src):
                pairs.add((s, d))
        return sorted(pairs)

    relations = {
        ("phenotype", "associated_with", "gene"): edges(n_ph, n_gene, 3),
        ("gene", "causes", "disease"): edges(n_gene, n_dis, 2),
        ("disease", "has_phenotype", "phenotype"): edges(n_dis, n_ph, 5),
        ("gene", "interacts_with", "gene"): [(a, b) for a, b in edges(n_gene, n_gene, 2)
                                              if a != b],
    }
    edge_index_dict = {}
    for (s, r, d), pairs in relations.items():
        ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        edge_index_dict[(s, r, d)] = ei
        edge_index_dict[(d, f"rev_{r}", s)] = ei.flip(0).contiguous()
    num_nodes = {"phenotype": n_ph, "gene": n_gene, "disease": n_dis}
    torch.manual_seed(seed)
    x_dict = {t: torch.randn(n, 8) for t, n in num_nodes.items()}
    torch.save(x_dict, data_dir / "node_features.pt")
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")
    (data_dir / "num_nodes.json").write_text(json.dumps(num_nodes))
    (data_dir / "kg.json").write_text(json.dumps({"synthetic": True, "seed": seed}))

    profiles = {d: {"phenotype_ids": [d % n_ph, (d + 3) % n_ph], "gene_ids": [d % n_gene]}
                for d in range(n_dis)}

    def samples(split, ids):
        return [{"patient_id": f"{split}-{d}", "phenotype_ids": list(profiles[d]["phenotype_ids"]),
                 "disease_id": d, "gene_ids": list(profiles[d]["gene_ids"]),
                 "candidate_disease_ids": sorted(ids)} for d in ids]

    train_ids, val_ids = [0, 1, 2, 3, 4, 5], [6, 7]
    write_generated_workspace(data_dir, train_ids=train_ids, val_ids=val_ids, profiles=profiles,
                              train_samples=samples("train", train_ids),
                              val_samples=samples("val", val_ids))
    return data_dir
