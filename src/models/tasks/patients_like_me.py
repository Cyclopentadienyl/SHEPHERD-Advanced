"""
Patients-like-me: cases bound to the view, ranked against a reference set.
=========================================================================
`docs/working/scorer-retraining/README.md` A9, F9, F11, E1–E5, U9;
`docs/working/PLAN_TEST_RESULTS.md` §8.4.

**Cases.** A patient's phenotypes are mapped to the view (dropped and counted
outside it); its declared causal genes are kept as the workspace's gene ids,
and those that name no single gene of the graph as the importer's normalised
identifiers (`causal_genes_unmapped`), so a gene outside the view, or outside
the graph, still identifies the gene two patients share. A pair is positive
when the two declared sets intersect (F9): the same gene of the graph, or the
same identifier outside it; an identifier outside the graph never matches a
graph gene (no alias is resolved). Its causal genes, candidate genes and
disease in the view are kept as well: training seeds them (U9). A case with no
phenotype in the view is excluded, in the state its truth gives it (E4): no
reliable truth without a declared causal gene, no other positive when no one
else in the cohort shares one, else input excluded with every same-gene patient
as a positive (E5). One without a declared causal gene has no reliable truth
and is neither an anchor nor a query with positives.

**Ranking (E1–E5).** Each query patient against every reference patient but
itself, the self excluded by stable case identity (`case_id`), also when the
query and reference lists are ordered differently; scores `−‖ẑ_a − ẑ_b‖²` of
the patient tower's vectors (A9, U10), exact, average ranks. States kept apart:
no other positive in the reference set, no reliable truth, input excluded;
none of them is a recall of 0. Same-gene patients excluded at binding are a
ranked query's relevant items it could not rank (E5), and a query whose
positives were all excluded is not scorable.

Module: src/models/tasks/patients_like_me.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from src.models.bundle.metrics import (
    NO_OTHER_POSITIVE,
    NO_RELIABLE_TRUTH,
    NOT_SCORABLE,
    RANKED,
    QueryOutcome,
    average_ranks,
)
from src.models.tasks.scoring import Binding, padded, patient_vectors


def _no_ids() -> Tensor:
    return torch.empty(0, dtype=torch.long)


@dataclass
class BoundPatient:
    case_id: str
    #: View ids of the patient's phenotypes in the view.
    phenotypes: Tensor
    #: The declared causal genes (empty: no reliable truth): the workspace's
    #: gene ids, and the normalised identifiers of those outside the graph.
    genes: FrozenSet[Union[int, str]]
    #: View ids of its causal genes and candidate genes in the view, and of
    #: its disease or -1: training seeds them (U9).
    truths: Tensor = field(default_factory=_no_ids)
    candidates: Tensor = field(default_factory=_no_ids)
    disease: int = -1


def _in_view(view: Any, node_type: str, ids: Optional[Sequence[int]]) -> Tensor:
    if not ids or node_type not in view.node_types:
        return _no_ids()
    mapped = view.view_ids(node_type, torch.tensor(list(ids), dtype=torch.long))
    return torch.unique(mapped[mapped >= 0])


def declared_genes(sample: Any) -> FrozenSet[Union[int, str]]:
    """A row's declared causal genes: the graph's gene ids and the identifiers
    that name no single gene of it (`causal_genes_unmapped`)."""
    return (frozenset(int(g) for g in (sample.causal_gene_ids or ()))
            | frozenset(getattr(sample, "causal_genes_unmapped", None) or ()))


def bind_patients(view: Any, samples: Sequence[Any]) -> Binding:
    """Map rows (`patient_id`, `phenotype_ids`, `causal_gene_ids`, and the
    optional `causal_genes_unmapped`, `gene_ids` and `disease_id`) to bound
    patients; a patient with no phenotype in the view is excluded, in its E4
    state."""
    cases: List[BoundPatient] = []
    excluded: List[Tuple[str, str]] = []
    counts = {"cases": len(samples), "phenotypes_outside_view": 0, "no_causal_gene": 0,
              "with_causal_genes_outside_graph": 0}
    seen = set()
    by_gene: Dict[Union[int, str], set] = {}
    for sample in samples:
        if sample.patient_id in seen:
            raise ValueError(f"case id {sample.patient_id!r} appears twice; self-exclusion "
                             "needs a stable, unique case identity (E5)")
        seen.add(sample.patient_id)
        for gene in declared_genes(sample):
            by_gene.setdefault(gene, set()).add(sample.patient_id)
        counts["with_causal_genes_outside_graph"] += int(
            bool(getattr(sample, "causal_genes_unmapped", None)))
    excluded_ids = set()
    for sample in samples:
        phen = view.view_ids("phenotype", torch.tensor(list(sample.phenotype_ids),
                                                       dtype=torch.long))
        counts["phenotypes_outside_view"] += int((phen < 0).sum())
        phen = torch.unique(phen[phen >= 0])
        if phen.numel() == 0:
            excluded.append((sample.patient_id, "no_phenotype_in_view"))
            excluded_ids.add(sample.patient_id)
            continue
        genes = declared_genes(sample)
        counts["no_causal_gene"] += int(not genes)
        disease = getattr(sample, "disease_id", None)
        in_view = _in_view(view, "disease", [] if disease is None else [int(disease)])
        cases.append(BoundPatient(sample.patient_id, phen, genes,
                                  _in_view(view, "gene", sample.causal_gene_ids),
                                  _in_view(view, "gene", getattr(sample, "gene_ids", None)),
                                  int(in_view[0]) if in_view.numel() else -1))

    def partners(case_id: str, genes: FrozenSet[Union[int, str]]) -> set:
        return set().union(*(by_gene[g] for g in genes)) - {case_id} if genes else set()

    states: Dict[str, str] = {}
    positives: Dict[str, int] = {}
    for sample in samples:
        if sample.patient_id not in excluded_ids:
            continue
        genes = declared_genes(sample)
        others = partners(sample.patient_id, genes)
        if not genes:
            states[sample.patient_id] = NO_RELIABLE_TRUTH
        elif not others:
            states[sample.patient_id] = NO_OTHER_POSITIVE
        else:
            positives[sample.patient_id] = len(others)
    unranked = {c.case_id: n for c in cases
                if (n := len(partners(c.case_id, c.genes) & excluded_ids))}
    counts.update({"bound": len(cases), "excluded": len(excluded)})
    for _, reason in excluded:
        counts[f"excluded_{reason}"] = counts.get(f"excluded_{reason}", 0) + 1
    return Binding(cases, excluded, counts, positives, states, unranked)


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
                  reference: Sequence[BoundPatient], *, chunk: int = 256,
                  unranked: Optional[Mapping[str, int]] = None) -> List[QueryOutcome]:
    """Each query's positives ranked among the reference patients but itself.
    `unranked`: a query's same-gene patients excluded from the reference at
    binding (`Binding.unranked_positives`, when the reference is the queries'
    own cohort), relevant items it could not rank (E5)."""
    unranked = unranked or {}
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
        missing = int(unranked.get(query.case_id, 0))
        if relevant == 0:
            outcomes.append(QueryOutcome(query.case_id, NOT_SCORABLE, n_positives=missing,
                                         reason="positives_excluded") if missing else
                            QueryOutcome(query.case_id, NO_OTHER_POSITIVE))
            continue
        keep = ~is_self[i]
        scores = head.scores(q_vec[i:i + 1], r_vec[keep])[0]
        ranks = average_ranks(scores.tolist())
        where = torch.nonzero(positive[i][keep]).flatten()
        outcomes.append(QueryOutcome(query.case_id, RANKED, [float(ranks[j]) for j in where],
                                     n_positives=relevant + missing,
                                     candidates=int(keep.sum())))
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


__all__ = ["BoundPatient", "anchor_groups", "bind_patients", "declared_genes",
           "patient_tower_vectors", "positive_matrix", "rank_patients", "self_matrix",
           "shares_gene"]
