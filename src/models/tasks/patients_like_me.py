"""
Patients-like-me: cases bound to the view, ranked against a reference set.
=========================================================================
`docs/working/scorer-retraining/README.md` A9, F9, F11, E1–E5, U9;
`docs/working/PLAN_TEST_RESULTS.md` §8.4.

**Cases.** A patient's phenotypes are mapped to the view (dropped and counted
outside it); its causal genes are kept as the workspace's gene ids, so a gene
outside the view still identifies the gene two patients share. A pair is
positive when the two causal-gene sets intersect (F9). A case with no phenotype
in the view is excluded; one without a causal gene has no reliable truth (E4)
and is neither an anchor nor a query with positives.

**Ranking (E1–E5).** Each query patient against every reference patient but
itself, the self excluded by stable case identity (`case_id`), also when the
query and reference lists are ordered differently; scores `−‖ẑ_a − ẑ_b‖²` of
the patient tower's vectors (A9, U10), exact, average ranks. States kept apart:
no other positive in the reference set, no reliable truth, input excluded;
none of them is a recall of 0.

Module: src/models/tasks/patients_like_me.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple

import torch
from torch import Tensor

from src.models.bundle.metrics import (
    NO_OTHER_POSITIVE,
    NO_RELIABLE_TRUTH,
    RANKED,
    QueryOutcome,
    average_ranks,
)
from src.models.tasks.scoring import Binding, padded, patient_vectors


@dataclass
class BoundPatient:
    case_id: str
    #: View ids of the patient's phenotypes in the view.
    phenotypes: Tensor
    #: The causal genes, as the workspace's gene ids (empty: no reliable truth).
    genes: FrozenSet[int]


def bind_patients(view: Any, samples: Sequence[Any]) -> Binding:
    """Map rows (`patient_id`, `phenotype_ids`, `causal_gene_ids`) to bound
    patients; a patient with no phenotype in the view is excluded."""
    cases: List[BoundPatient] = []
    excluded: List[Tuple[str, str]] = []
    counts = {"cases": len(samples), "phenotypes_outside_view": 0, "no_causal_gene": 0}
    seen = set()
    for sample in samples:
        if sample.patient_id in seen:
            raise ValueError(f"case id {sample.patient_id!r} appears twice; self-exclusion "
                             "needs a stable, unique case identity (E5)")
        seen.add(sample.patient_id)
        phen = view.view_ids("phenotype", torch.tensor(list(sample.phenotype_ids),
                                                       dtype=torch.long))
        counts["phenotypes_outside_view"] += int((phen < 0).sum())
        phen = torch.unique(phen[phen >= 0])
        if phen.numel() == 0:
            excluded.append((sample.patient_id, "no_phenotype_in_view"))
            continue
        genes = frozenset(int(g) for g in (sample.causal_gene_ids or ()))
        counts["no_causal_gene"] += int(not genes)
        cases.append(BoundPatient(sample.patient_id, phen, genes))
    counts.update({"bound": len(cases), "excluded": len(excluded)})
    for _, reason in excluded:
        counts[f"excluded_{reason}"] = counts.get(f"excluded_{reason}", 0) + 1
    return Binding(cases, excluded, counts)


def shares_gene(a: BoundPatient, b: BoundPatient) -> bool:
    return bool(a.genes & b.genes)


def positive_matrix(queries: Sequence[BoundPatient], reference: Sequence[BoundPatient]) -> Tensor:
    """`[Q, R]`: the two patients' causal-gene sets intersect, and they are
    not the same case (by case id)."""
    out = torch.zeros(len(queries), len(reference), dtype=torch.bool)
    for i, q in enumerate(queries):
        for j, r in enumerate(reference):
            out[i, j] = q.case_id != r.case_id and shares_gene(q, r)
    return out


def self_matrix(queries: Sequence[BoundPatient], reference: Sequence[BoundPatient]) -> Tensor:
    index = {c.case_id: j for j, c in enumerate(reference)}
    out = torch.zeros(len(queries), len(reference), dtype=torch.bool)
    for i, q in enumerate(queries):
        if q.case_id in index:
            out[i, index[q.case_id]] = True
    return out


@torch.no_grad()
def patient_tower_vectors(bundle: Any, z: Tensor, patients: Sequence[BoundPatient], *,
                          chunk: int = 256) -> Tensor:
    """Each patient's projected vector (`patient_vectors`), full-view `z`."""
    rows = []
    for start in range(0, len(patients), chunk):
        index, mask = padded(patients[start:start + chunk])
        rows.append(patient_vectors(bundle, z, index.to(z.device), mask.to(z.device)).cpu())
    return torch.cat(rows) if rows else torch.empty(0, 0)


def rank_patients(bundle: Any, z: Tensor, queries: Sequence[BoundPatient],
                  reference: Sequence[BoundPatient], *, chunk: int = 256) -> List[QueryOutcome]:
    """Each query's positives ranked among the reference patients but itself."""
    head = bundle.task_head
    q_vec = patient_tower_vectors(bundle, z, queries, chunk=chunk)
    r_vec = q_vec if queries is reference else patient_tower_vectors(bundle, z, reference,
                                                                      chunk=chunk)
    is_self = self_matrix(queries, reference)
    positive = positive_matrix(queries, reference)
    outcomes: List[QueryOutcome] = []
    for i, query in enumerate(queries):
        if not query.genes:
            outcomes.append(QueryOutcome(query.case_id, NO_RELIABLE_TRUTH))
            continue
        relevant = int(positive[i].sum())
        if relevant == 0:
            outcomes.append(QueryOutcome(query.case_id, NO_OTHER_POSITIVE))
            continue
        keep = ~is_self[i]
        scores = head.scores(q_vec[i:i + 1], r_vec[keep])[0]
        ranks = average_ranks(scores.tolist())
        where = torch.nonzero(positive[i][keep]).flatten()
        outcomes.append(QueryOutcome(query.case_id, RANKED, [float(ranks[j]) for j in where],
                                     n_positives=relevant, candidates=int(keep.sum())))
    return outcomes


def anchor_groups(cases: Sequence[BoundPatient]) -> Tuple[List[int], Dict[int, List[int]]]:
    """F11: the indices of patients with at least two other patients sharing a
    causal gene, and each patient's same-gene partners (ascending)."""
    by_gene: Dict[int, List[int]] = {}
    for i, case in enumerate(cases):
        for gene in case.genes:
            by_gene.setdefault(gene, []).append(i)
    partners: Dict[int, List[int]] = {}
    for i, case in enumerate(cases):
        others = sorted({j for g in case.genes for j in by_gene[g] if j != i})
        partners[i] = others
    anchors = [i for i in range(len(cases)) if len(partners[i]) >= 2]
    return anchors, partners


__all__ = ["BoundPatient", "anchor_groups", "bind_patients", "patient_tower_vectors",
           "positive_matrix", "rank_patients", "self_matrix", "shares_gene"]
