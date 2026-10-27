"""
The gene task's score: EMBSIM mixed with the fixed-bound SPLSIM, Eq 10 augmentation.
===================================================================================
`docs/working/scorer-retraining/README.md` A8, C9, U12, F9, F10 and §6; the
paper's Eq 10–11 (Alsentzer et al. 2025, Methods, "gene augmentation").

**SPLSIM (§6).** `S = 1 − 2(d̄ − lower)/(upper − lower)`, `d̄` the mean hop
distance from the gene to the patient's phenotypes over the whole view
(`src.kg.graph_view.exact_distances`), bounds from the spec, no clipping. A
pair with any unreachable distance has no `d̄`: it is undefined, never scored
with a cap (§6 rule 7); the caller counts it.

**Augmentation (F10, the paper's Eq 10–11), after the task projection.** A
candidate gene's K neighbours are the genes sharing the most phenotypes with it
through the view's gene→phenotype relations (a raw count, the paper's
`|P_g ∩ P_h|`), self excluded, ties by view id, padding where fewer than K
share any. Weights are the counts over their sum (padding outside it). The gene
becomes `(1 − θ_g)·ĝ + θ_g·Σ w_k ĝ_k` with the paper's degree gate
`θ_g = ω·exp(−ω·|N_g|) + 0.2`, `|N_g|` the gene's degree in the view (its
out-edges, each fact's reverse twin included). A gene with no neighbour keeps
its own vector and says so. The same rule runs in training, validation,
measurement and serving. (The source applies its gate before the projection,
in training only; README F10.)

**The score** is `η·EMBSIM + (1 − η)·SPLSIM` (A8, U12); the multi-similarity
loss is computed on it.

Module: src/models/tasks/gene.py
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

#: The degree gate's constant term (the paper's Eq 11).
GATE_FLOOR = 0.2
SP_RULE = ("S = 1 - 2(mean_distance - lower)/(upper - lower); mean over the patient's "
           "phenotypes; exact hops on the whole view; no clipping; undefined when any "
           "distance is unreachable")
AUGMENTATION_RULE = ("after the projection; K genes sharing the most phenotypes (view "
                     "gene->phenotype relations), self excluded, ties by view id; weights "
                     "count/sum; theta = omega*exp(-omega*degree) + 0.2, degree = view out-edges")


def splsim(mean_distance: Tensor, lower: float, upper: float) -> Tensor:
    """§6's fixed-bound similarity; closer is higher, nothing clipped."""
    return 1.0 - 2.0 * (mean_distance - lower) / (upper - lower)


def mean_distances(distances: Tensor, phenotype_rows: Tensor, phenotype_mask: Tensor,
                   gene_cols: Tensor) -> Tuple[Tensor, Tensor]:
    """`d̄` for each patient's candidates and whether it is defined.

    `distances` is `[R, G]` hops (-1 unreachable); `phenotype_rows` `[B, P]`
    its rows for each patient's phenotypes (masked by `phenotype_mask`);
    `gene_cols` `[B, M]` its columns for each candidate. Returns `[B, M]` means
    and `[B, M]` defined flags."""
    d = distances[phenotype_rows.unsqueeze(-1), gene_cols.unsqueeze(1)]
    real = phenotype_mask.unsqueeze(-1)
    undefined = ((d < 0) & real).any(dim=1)
    total = (d.clamp(min=0).to(torch.float32) * real).sum(dim=1)
    count = phenotype_mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
    return total / count, ~undefined


@dataclass
class NeighbourTable:
    """Each view node's augmentation neighbours (rows of non-genes are empty)."""

    #: `[N, K]` view ids, -1 for padding.
    neighbours: Tensor
    #: `[N, K]` weights summing to 1 over the real neighbours, 0 for padding.
    weights: Tensor
    #: `[N]` degree in the view.
    degree: Tensor
    record: Dict[str, Any] = field(default_factory=dict)


def neighbour_table(view: Any, k: int, *, chunk_size: int = 1024) -> NeighbourTable:
    """The F10 neighbour table from the view's gene→phenotype relations. The
    incidence is sparse; `chunk_size` genes' shared counts are dense at a time."""
    import numpy as np
    from scipy.sparse import csr_matrix

    n = int(view.num_nodes)
    genes = view.type_nodes("gene")
    relations = [i for i, key in enumerate(view.relations)
                 if key.split("|")[0] == "gene" and key.split("|")[-1] == "phenotype"]
    keep = torch.isin(view.rel, torch.tensor(relations, dtype=torch.long))
    src, dst = view.src[keep], view.dst[keep]
    phenotypes = torch.unique(dst)
    gene_pos = torch.full((n,), -1, dtype=torch.long)
    gene_pos[genes] = torch.arange(genes.numel())
    phen_pos = torch.full((n,), -1, dtype=torch.long)
    phen_pos[phenotypes] = torch.arange(phenotypes.numel())
    incidence = csr_matrix(
        (np.ones(src.numel(), dtype=np.float32), (gene_pos[src].numpy(), phen_pos[dst].numpy())),
        shape=(genes.numel(), phenotypes.numel()))
    incidence.data[:] = 1.0  # a gene–phenotype pair counts once, whatever its relations
    transposed = incidence.T.tocsr()
    neighbours = torch.full((n, k), -1, dtype=torch.long)
    weights = torch.zeros(n, k, dtype=torch.float32)
    for start in range(0, genes.numel(), chunk_size):
        rows = torch.arange(start, min(start + chunk_size, genes.numel()))
        shared = torch.from_numpy(
            (incidence[rows.numpy()] @ transposed).toarray().astype(np.float32))
        shared[torch.arange(rows.numel()), rows] = 0.0
        # Stable sort on the negated count: ties keep ascending view id.
        order = torch.sort(-shared, dim=1, stable=True).indices[:, :k]
        counts = torch.gather(shared, 1, order)
        real = counts > 0
        total = counts.sum(dim=1, keepdim=True)
        w = torch.where(real, counts / total.clamp(min=1.0), torch.zeros_like(counts))
        ids = torch.where(real, genes[order], torch.full_like(order, -1))
        neighbours[genes[rows], :ids.size(1)] = ids
        weights[genes[rows], :w.size(1)] = w
    degree = torch.bincount(view.src, minlength=n).to(torch.float32)
    digest = hashlib.sha256()
    for t in (neighbours, weights, degree):
        digest.update(t.contiguous().numpy().tobytes())
    record = {"rule": AUGMENTATION_RULE, "k": int(k),
              "relations": [view.relations[i] for i in relations],
              "genes_without_neighbour": int((neighbours[genes, 0] < 0).sum()),
              "digest": digest.hexdigest()}
    return NeighbourTable(neighbours, weights, degree, record)


def augment(genes: Tensor, neighbours: Tensor, weights: Tensor, degree: Tensor,
            omega: float) -> Tuple[Tensor, Tensor]:
    """Eq 10–11 on projected vectors: `genes` `[..., W]`, `neighbours`
    `[..., K, W]`, `weights` `[..., K]`, `degree` `[...]`. Returns the
    augmented vectors and whether each gene had a neighbour."""
    has = weights.sum(dim=-1) > 0
    aggregate = (weights.unsqueeze(-1) * neighbours).sum(dim=-2)
    theta = (omega * torch.exp(-omega * degree) + GATE_FLOOR) * has.to(genes.dtype)
    theta = theta.unsqueeze(-1)
    return (1.0 - theta) * genes + theta * aggregate, has


def gene_list_scores(bundle: Any, z: Tensor, phenotype_index: Tensor, phenotype_mask: Tensor,
                     gene_index: Tensor, neighbour_index: Tensor, neighbour_weight: Tensor,
                     degree: Tensor, sp_similarity: Tensor) -> Tensor:
    """`[B, M]` gene scores of each patient's own list. Indices are rows of
    `z` (seed positions in training, view ids on the full view);
    `neighbour_index` is `[B, M, K]` (any row where its weight is 0)."""
    from src.models.tasks.scoring import patient_vectors

    head = bundle.task_head
    spec = head.spec.gene
    patients = patient_vectors(bundle, z, phenotype_index, phenotype_mask)
    genes = head.project_candidates(z[gene_index])
    if spec.augmentation:
        near = head.project_candidates(z[neighbour_index.clamp(min=0)])
        genes, _ = augment(genes, near, neighbour_weight, degree, spec.augmentation_omega)
    embsim = head.list_scores(patients, genes)
    return spec.eta * embsim + (1.0 - spec.eta) * sp_similarity


__all__ = [
    "AUGMENTATION_RULE",
    "GATE_FLOOR",
    "NeighbourTable",
    "SP_RULE",
    "augment",
    "gene_list_scores",
    "mean_distances",
    "neighbour_table",
    "splsim",
]
