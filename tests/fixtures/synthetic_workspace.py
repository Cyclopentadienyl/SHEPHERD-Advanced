"""
A synthetic data directory and checkpoint, built so sampling randomness cannot
change the answer.
======================================================================
Neither the frozen evaluator nor the dataloader seeds anything: neighbour
sampling uses `random.sample` / `random.choice` (`src/kg/data_loader.py:220,263`)
and negatives use `np.random.choice` / `random.randint` (`:534-597`). Two
processes running the same command can therefore build different subgraphs.

Rather than seed the RNG — which would mean reaching into production code — the
graph is shaped so the draw cannot matter:

  - **every node's degree is at or below the neighbour-sampling limit**
    (`num_neighbors = [15, 10, 5]`), so `random.sample` never has a choice to
    make;
  - **every gene links every phenotype to every disease**, so the 2-hop expansion
    from any disease seed reaches all diseases. Whichever negatives are drawn,
    the candidate universe is the same set.

`assert_candidate_universe_is_stable` checks that empirically over several seeds
rather than trusting the argument, because the argument depends on sampling
internals that are not this test's to police.

Module: tests/fixtures/synthetic_workspace.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Small enough that every degree sits under the sampling limits, and complete
# enough that the 2-hop disease universe is the whole disease set.
N_PHENOTYPES = 3
N_GENES = 2
N_DISEASES = 4
N_SAMPLES = 6
FEATURE_DIM = 8
HIDDEN_DIM = 8


def _graph() -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], Dict[str, int]]:
    torch.manual_seed(20260818)
    x_dict = {
        "phenotype": torch.randn(N_PHENOTYPES, FEATURE_DIM),
        "gene": torch.randn(N_GENES, FEATURE_DIM),
        "disease": torch.randn(N_DISEASES, FEATURE_DIM),
    }

    ph_gene = torch.tensor(
        [[p, g] for p in range(N_PHENOTYPES) for g in range(N_GENES)], dtype=torch.long
    ).t()
    gene_dis = torch.tensor(
        [[g, d] for g in range(N_GENES) for d in range(N_DISEASES)], dtype=torch.long
    ).t()

    edge_index_dict = {
        ("phenotype", "associated_with", "gene"): ph_gene,
        ("gene", "causes", "disease"): gene_dis,
    }
    num_nodes_dict = {
        "phenotype": N_PHENOTYPES,
        "gene": N_GENES,
        "disease": N_DISEASES,
    }
    return x_dict, edge_index_dict, num_nodes_dict


def _samples() -> List[dict]:
    """Deterministic and hand-written — no RNG, so sample order is fixed."""
    return [
        {"patient_id": f"P{i:02d}",
         "phenotype_ids": [i % N_PHENOTYPES, (i + 1) % N_PHENOTYPES],
         "disease_id": i % N_DISEASES}
        for i in range(N_SAMPLES)
    ]


def build_workspace(root: Path, split: str = "test") -> Tuple[Path, Path]:
    """Write a data directory and a checkpoint. Returns ``(data_dir, checkpoint)``.

    The layout is exactly what `scripts/evaluate_model.py:164-223` reads, so the
    frozen oracle and the new harness can be pointed at the same fixture.
    """
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    x_dict, edge_index_dict, num_nodes_dict = _graph()
    torch.save(x_dict, data_dir / "node_features.pt")
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")
    (data_dir / "num_nodes.json").write_text(json.dumps(num_nodes_dict))
    (data_dir / f"{split}_samples.json").write_text(json.dumps(_samples()))

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
            "metadata": metadata,
            "in_channels_dict": in_channels_dict,
            "epoch": 1,
            "logs": {"val_mrr": 0.0},
        },
        checkpoint_path,
    )
    return data_dir, checkpoint_path


def assert_candidate_universe_is_stable(data_dir: Path, split: str, seeds=(0, 1, 7, 99)) -> None:
    """The fixture's precondition, checked rather than assumed.

    If different seeds produced different candidate sets, a comparison between two
    processes would be measuring the sampler rather than the scorer — and it would
    fail intermittently, which is worse than failing.
    """
    from src.kg.data_loader import (
        DataLoaderConfig,
        DiagnosisSample,
        create_diagnosis_dataloader,
    )

    raw = json.loads((data_dir / f"{split}_samples.json").read_text())
    samples = [
        DiagnosisSample(
            patient_id=s["patient_id"],
            phenotype_ids=s["phenotype_ids"],
            disease_id=s["disease_id"],
        )
        for s in raw
    ]
    graph_data = {
        "x_dict": torch.load(data_dir / "node_features.pt", weights_only=True),
        "edge_index_dict": torch.load(data_dir / "edge_indices.pt", weights_only=True),
        "num_nodes_dict": json.loads((data_dir / "num_nodes.json").read_text()),
    }

    observed = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader = create_diagnosis_dataloader(
            samples=samples,
            graph_data=graph_data,
            config=DataLoaderConfig(batch_size=3, num_workers=0, shuffle=False),
        )
        observed.append(
            [batch["original_indices"]["disease"].tolist() for batch in loader]
        )

    first = observed[0]
    for seed, run in zip(seeds[1:], observed[1:]):
        assert run == first, (
            f"the candidate universe moved under seed {seed}: {run} != {first}. "
            "The fixture is not randomness-insensitive and any comparison built on "
            "it would be measuring the sampler."
        )
