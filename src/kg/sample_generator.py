"""
Training Sample Generator
==========================
Generates simulated patient training samples from a KnowledgeGraph.

Each sample represents a "patient" with:
  - A set of observed phenotypes (from the KG)
  - A correct disease diagnosis (ground truth)
  - Optional: candidate genes

The generator traverses disease->phenotype and disease->gene edges
in the KG to create realistic training data for the diagnosis model.

Output format matches what scripts/train_model.py expects:
  [{"patient_id": str, "phenotype_ids": [int], "disease_id": int}, ...]
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def generate_training_samples(
    kg: KnowledgeGraph,
    num_train: int = 5000,
    num_val: int = 1000,
    min_phenotypes: int = 2,
    max_phenotypes: int = 15,
    phenotype_drop_rate: float = 0.3,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate training and validation samples from a KnowledgeGraph.

    For each sample:
      1. Pick a random disease that has at least `min_phenotypes` phenotypes
      2. Randomly drop some phenotypes (simulating incomplete observation)
      3. Record the disease index as ground truth
      4. Optionally collect associated genes

    Args:
        kg: KnowledgeGraph instance with nodes and edges loaded.
        num_train: Number of training samples.
        num_val: Number of validation samples.
        min_phenotypes: Minimum phenotypes a disease must have to be eligible.
        max_phenotypes: Maximum phenotypes to include per sample.
        phenotype_drop_rate: Fraction of phenotypes to randomly drop per sample.
        seed: Random seed for reproducibility.
        output_dir: If provided, save train_samples.json and val_samples.json.

    Returns:
        (train_samples, val_samples) as lists of dicts.
    """
    rng = random.Random(seed)
    node_mapping = kg.get_node_id_mapping()

    disease_profiles = _build_disease_profiles(kg, node_mapping)

    # Filter diseases with enough phenotypes
    eligible_diseases = [
        (disease_idx, profile)
        for disease_idx, profile in disease_profiles.items()
        if len(profile["phenotype_ids"]) >= min_phenotypes
    ]

    if not eligible_diseases:
        logger.warning(
            "No diseases with enough phenotypes found. "
            f"Need >= {min_phenotypes} phenotypes per disease."
        )
        return [], []

    logger.info(
        f"Found {len(eligible_diseases)} eligible diseases "
        f"(with >= {min_phenotypes} phenotypes)"
    )

    total = num_train + num_val
    samples = _generate_samples(
        eligible_diseases=eligible_diseases,
        total=total,
        min_phenotypes=min_phenotypes,
        max_phenotypes=max_phenotypes,
        phenotype_drop_rate=phenotype_drop_rate,
        rng=rng,
    )

    rng.shuffle(samples)
    train_samples = samples[:num_train]
    val_samples = samples[num_train:]

    logger.info(
        f"Generated {len(train_samples)} train, {len(val_samples)} val samples"
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "train_samples.json", "w") as f:
            json.dump(train_samples, f)
        with open(output_dir / "val_samples.json", "w") as f:
            json.dump(val_samples, f)
        logger.info(f"Samples saved to {output_dir}")

    return train_samples, val_samples


def _build_disease_profiles(
    kg: KnowledgeGraph,
    node_mapping: Dict[str, Dict[str, int]],
) -> Dict[int, Dict[str, Any]]:
    """
    Build a profile for each disease: its associated phenotypes and genes.

    Traverses edges to find:
      - Phenotypes linked to each disease (PHENOTYPE_OF_DISEASE, GENE_HAS_PHENOTYPE via shared gene)
      - Genes linked to each disease (GENE_ASSOCIATED_WITH_DISEASE)
    """
    disease_mapping = node_mapping.get("disease", {})
    phenotype_mapping = node_mapping.get("phenotype", {})
    gene_mapping = node_mapping.get("gene", {})

    # disease_idx -> {phenotype_ids: set, gene_ids: set}
    profiles: Dict[int, Dict[str, Set[int]]] = {}
    for d_idx in disease_mapping.values():
        profiles[d_idx] = {"phenotype_ids": set(), "gene_ids": set()}

    # Reverse lookups: node_id_str -> idx
    disease_strs = {v: k for k, v in disease_mapping.items()}
    gene_strs = {v: k for k, v in gene_mapping.items()}

    # Two-pass edge traversal:
    # Pass 1: collect direct edges (phenotype-disease, gene-disease)
    gene_phenotype_edges: List[Tuple[int, int]] = []

    for edge in kg._edges:
        src_str = str(edge.source_id)
        tgt_str = str(edge.target_id)
        et = edge.edge_type.value

        if et == "phenotype_of_disease":
            pheno_idx = phenotype_mapping.get(src_str)
            disease_idx = disease_mapping.get(tgt_str)
            if pheno_idx is not None and disease_idx is not None:
                profiles[disease_idx]["phenotype_ids"].add(pheno_idx)

        elif et == "gene_associated_with_disease":
            gene_idx = gene_mapping.get(src_str)
            disease_idx = disease_mapping.get(tgt_str)
            if gene_idx is not None and disease_idx is not None:
                profiles[disease_idx]["gene_ids"].add(gene_idx)

        elif et == "gene_has_phenotype":
            gene_idx = gene_mapping.get(src_str)
            pheno_idx = phenotype_mapping.get(tgt_str)
            if gene_idx is not None and pheno_idx is not None:
                gene_phenotype_edges.append((gene_idx, pheno_idx))

    # Pass 2: propagate gene-phenotype edges to diseases via reverse index
    gene_to_diseases: Dict[int, Set[int]] = {}
    for d_idx, prof in profiles.items():
        for g_idx in prof["gene_ids"]:
            gene_to_diseases.setdefault(g_idx, set()).add(d_idx)

    for gene_idx, pheno_idx in gene_phenotype_edges:
        for d_idx in gene_to_diseases.get(gene_idx, ()):
            profiles[d_idx]["phenotype_ids"].add(pheno_idx)

    return {
        d_idx: {
            "phenotype_ids": list(prof["phenotype_ids"]),
            "gene_ids": list(prof["gene_ids"]),
        }
        for d_idx, prof in profiles.items()
    }


def _generate_samples(
    eligible_diseases: List[Tuple[int, Dict[str, Any]]],
    total: int,
    min_phenotypes: int,
    max_phenotypes: int,
    phenotype_drop_rate: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate simulated patient samples."""
    samples = []

    for i in range(total):
        disease_idx, profile = rng.choice(eligible_diseases)
        all_phenos = profile["phenotype_ids"]

        # Randomly drop phenotypes
        n_keep = max(
            min_phenotypes,
            int(len(all_phenos) * (1.0 - phenotype_drop_rate)),
        )
        n_keep = min(n_keep, max_phenotypes, len(all_phenos))

        selected_phenos = rng.sample(all_phenos, n_keep)

        sample: Dict[str, Any] = {
            "patient_id": f"sim_patient_{i:06d}",
            "phenotype_ids": selected_phenos,
            "disease_id": disease_idx,
        }

        if profile["gene_ids"]:
            sample["gene_ids"] = profile["gene_ids"]

        samples.append(sample)

    return samples
