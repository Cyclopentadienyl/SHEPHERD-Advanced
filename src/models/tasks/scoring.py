"""
The one task scorer: cases bound to the view, ranked against every candidate.
============================================================================
`docs/working/scorer-retraining/README.md` §3.1 ("all three share the exact
chunked scorer and ranker and one metrics module"), U6, F7, E1–E5, §2.8.

Training validation (`src.training.finetune_task`), task measurement and
serving all bind cases and rank them here, so no consumer scores a patient its
own way.

- **Binding (F7).** A case's phenotypes and truths are mapped to view ids. A
  phenotype outside the view is dropped and counted; a case whose truth is
  outside the view, or with no phenotype left, is excluded and counted, never
  dropped silently.
- **Node vectors (§2.8).** The full-view chunked rule, over the edges the
  caller passes (the whole view for a finetuned model, G4).
- **Ranking (U6, E3).** Every candidate of the task, exactly, in chunks of
  cases; average ranks; the positives' ranks as `QueryOutcome`s for the one
  metrics module.

Module: src/models/tasks/scoring.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import torch
from torch import Tensor

from src.models.bundle.metrics import (
    INPUT_EXCLUDED,
    NO_RELIABLE_TRUTH,
    RANKED,
    QueryOutcome,
    average_ranks,
)


@dataclass
class BoundCase:
    case_id: str
    #: View ids of the case's phenotypes in the view.
    phenotypes: Tensor
    #: View ids of the case's truths.
    truths: Tensor


@dataclass
class Binding:
    cases: List[Any]
    excluded: List[Tuple[str, str]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    #: An excluded case's assessable positives (the truths it lists), where a
    #: task knows more than one (E5's conservative denominator); 1 otherwise.
    excluded_positives: Dict[str, int] = field(default_factory=dict)
    #: Excluded cases that record no truth at all (E4's no reliable truth).
    no_truth: Set[str] = field(default_factory=set)

    def excluded_outcomes(self) -> List[QueryOutcome]:
        """Each excluded case as a unit with its reason (E4): `NO_RELIABLE_TRUTH`
        when it records no truth, which stays in coverage but never reads as a
        0; else `INPUT_EXCLUDED`, which stays in coverage and the conservative
        reading (E5)."""
        return [QueryOutcome(case_id, NO_RELIABLE_TRUTH, reason=reason)
                if case_id in self.no_truth else
                QueryOutcome(case_id, INPUT_EXCLUDED,
                             n_positives=self.excluded_positives.get(case_id, 1), reason=reason)
                for case_id, reason in self.excluded]


def bind_disease_cases(view: Any, samples: Sequence[Any]) -> Binding:
    """Map samples (per-type local ids, `patient_id`, `phenotype_ids`,
    `disease_id`) to view ids (F7)."""
    cases: List[BoundCase] = []
    excluded: List[Tuple[str, str]] = []
    dropped_phenotypes = 0
    for sample in samples:
        phen = view.view_ids("phenotype",
                             torch.tensor(list(sample.phenotype_ids), dtype=torch.long))
        kept = phen[phen >= 0]
        dropped_phenotypes += int((phen < 0).sum())
        truth = view.view_ids("disease", torch.tensor([int(sample.disease_id)]))
        if int(truth[0]) < 0:
            excluded.append((sample.patient_id, "truth_outside_view"))
            continue
        if kept.numel() == 0:
            excluded.append((sample.patient_id, "no_phenotype_in_view"))
            continue
        cases.append(BoundCase(sample.patient_id, torch.unique(kept), truth))
    counts = {"cases": len(samples), "bound": len(cases), "excluded": len(excluded),
              "phenotypes_outside_view": dropped_phenotypes}
    for _, reason in excluded:
        counts[f"excluded_{reason}"] = counts.get(f"excluded_{reason}", 0) + 1
    return Binding(cases, excluded, counts)


def patient_vectors(bundle: Any, z: Tensor, phenotype_index: Tensor,
                    phenotype_mask: Tensor) -> Tensor:
    """Phenotype vectors → patient encoder → the patient tower (A6, C5)."""
    pooled, _ = bundle.patient_encoder(z[phenotype_index], phenotype_mask)
    return bundle.task_head.patient_tower(pooled)


def padded(cases: Sequence[BoundCase]) -> Tuple[Tensor, Tensor]:
    """`[B, P]` phenotype view ids of `cases`, padded with 0, and the mask."""
    width = max(c.phenotypes.numel() for c in cases)
    index = torch.zeros(len(cases), width, dtype=torch.long)
    mask = torch.zeros(len(cases), width, dtype=torch.bool)
    for i, c in enumerate(cases):
        index[i, :c.phenotypes.numel()] = c.phenotypes
        mask[i, :c.phenotypes.numel()] = True
    return index, mask


@torch.no_grad()
def full_view_vectors(bundle: Any, edges: Tensor, relations: Tensor, device: torch.device,
                      chunk_size: int = 65536) -> Tensor:
    """Every view node's vector by the full-view rule (§2.8), in evaluation mode."""
    bundle.eval()
    return bundle.node_vectors_full(edges.to(device), edge_type=relations.to(device),
                                    chunk_size=chunk_size)


@torch.no_grad()
def score_cases(bundle: Any, z: Tensor, cases: Sequence[BoundCase], candidates: Tensor,
                *, case_chunk: int = 256) -> Tensor:
    """`[len(cases), len(candidates)]` task scores, exactly (no approximation)."""
    head = bundle.task_head
    projected = head.project_candidates(z[candidates.to(z.device)])
    rows = []
    for start in range(0, len(cases), case_chunk):
        index, mask = padded(cases[start:start + case_chunk])
        p = patient_vectors(bundle, z, index.to(z.device), mask.to(z.device))
        rows.append(head.scores(p, projected).cpu())
    return torch.cat(rows) if rows else torch.empty(0, candidates.numel())


def rank_cases(bundle: Any, z: Tensor, cases: Sequence[BoundCase], candidates: Tensor,
               *, case_chunk: int = 256) -> List[QueryOutcome]:
    """Each case's positives ranked among every candidate, average ranks (E3)."""
    scores = score_cases(bundle, z, cases, candidates, case_chunk=case_chunk)
    outcomes: List[QueryOutcome] = []
    for i, case in enumerate(cases):
        ranks = average_ranks(scores[i].tolist())
        where = torch.nonzero(torch.isin(candidates, case.truths)).flatten()
        outcomes.append(QueryOutcome(case.case_id, RANKED, [float(ranks[j]) for j in where],
                                     candidates=int(candidates.numel())))
    return outcomes


__all__ = [
    "Binding",
    "BoundCase",
    "bind_disease_cases",
    "full_view_vectors",
    "padded",
    "patient_vectors",
    "rank_cases",
    "score_cases",
]
