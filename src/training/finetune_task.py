"""
The finetune stage: a task head trained jointly with link prediction.
====================================================================
`docs/working/scorer-retraining/README.md` A6, A7, A10, C3, F5, F7, U6–U9,
G4, E1–E5. The disease task here; the gene and patients-like-me tasks add
their candidate rules to the same sampler and step.

**Binding (F7)** and **ranking (U6)** are the shared task scorer's
(`src.models.tasks.scoring`), which measurement and serving use too.

**Batches (U9).** Each epoch takes the bound training cases in a seeded order,
`batch_size` a batch. Seeds are the batch's phenotypes and truths, the task's
candidates (disease: `candidate_sample_size` random diseases of the view plus
the truths, shared by the batch; `up:hparams.py:101`) and `sparse_sample_size`
random gene, phenotype and disease nodes. LP sources are the seeds of those
three types (U7); their draws, positives, negatives and the F5 filter are the
base sampler's (`src.training.lp_task`), over the training edges (G4).

**Step (A7, C3).** Phenotype vectors pass the patient encoder and the patient
tower; candidates their tower; the NCA loss over the shared candidates is the
task term. The loss is `task_loss_weight · task + lp_loss_weight · lp`; a step
without an LP pair keeps the weighted task term and counts the absence.

**Validation (U6, G4, E1–E5).** Node vectors by full-view chunked inference
over the whole view; every disease of the view is a candidate; ranks are
average ranks; pooled MRR is the selection metric, with macro recall, coverage
and the conservative reading beside it.

Module: src/training/finetune_task.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence

import torch
from torch import Tensor

from src.models.bundle.metrics import INPUT_EXCLUDED, QueryOutcome, ranking_report
from src.models.tasks.losses import nca_loss
from src.models.tasks.scoring import (
    Binding,
    BoundCase,
    bind_disease_cases,
    full_view_vectors,
    patient_vectors,
    rank_cases,
)
from src.training.lp_task import BaseLPSampler, LPBatch, StepResult

#: The finetune selection metric (E1), as the Trainer logs it.
FINETUNE_SELECTION_METRIC = "val_pooled_mrr"
#: Node types whose seeds are LP sources (U7: the source's seed-type rule).
LP_SOURCE_TYPES = ("gene", "phenotype", "disease")


@dataclass
class TaskBatch:
    """One finetune step: the LP subgraph and the task's patients."""

    lp: LPBatch
    #: `[B, P]` seed indices of each patient's phenotypes, padded with 0.
    phenotype_index: Tensor
    phenotype_mask: Tensor
    #: Seed indices of the shared candidates, and `[B, M]` truth membership.
    candidate_index: Tensor
    positive: Tensor
    counts: Dict[str, int] = field(default_factory=dict)


class FinetuneSampler:
    """Patient batches for the disease task (U9), reusing the base sampler's
    LP draws over the training edges."""

    def __init__(self, view: Any, cases: Sequence[BoundCase], lp: BaseLPSampler, *,
                 batch_size: int, candidate_sample_size: int, sparse_sample_size: int, seed: int):
        if not cases:
            raise ValueError("no bound training case; the finetune has nothing to train on")
        self.view = view
        self.cases = list(cases)
        self.lp = lp
        self.batch_size = int(batch_size)
        self.candidate_sample_size = int(candidate_sample_size)
        self.sparse_sample_size = int(sparse_sample_size)
        self.seed = int(seed)
        self.epoch = 0
        self.diseases = view.type_nodes("disease")
        self.sparse_pool = torch.cat([view.type_nodes(t) for t in LP_SOURCE_TYPES])
        types = [view.node_types.index(t) for t in LP_SOURCE_TYPES if t in view.node_types]
        self._lp_type = torch.zeros(len(view.node_types), dtype=torch.bool)
        self._lp_type[types] = True

    def __len__(self) -> int:
        return (len(self.cases) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[TaskBatch]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(len(self.cases), generator=generator).tolist()
        for start in range(0, len(order), self.batch_size):
            yield self.batch([self.cases[i] for i in order[start:start + self.batch_size]],
                             generator)

    def batch(self, cases: Sequence[BoundCase], generator: torch.Generator) -> TaskBatch:
        truths = torch.unique(torch.cat([c.truths for c in cases]))
        k = min(self.candidate_sample_size, self.diseases.numel())
        sampled = self.diseases[torch.randperm(self.diseases.numel(), generator=generator)[:k]]
        candidates = torch.unique(torch.cat([sampled, truths]))
        s = min(self.sparse_sample_size, self.sparse_pool.numel())
        sparse = self.sparse_pool[torch.randperm(self.sparse_pool.numel(), generator=generator)[:s]]
        phenotypes = torch.cat([c.phenotypes for c in cases])
        task_nodes = torch.unique(torch.cat([phenotypes, candidates, sparse]))
        sources = task_nodes[self._lp_type[self.view.node_type[task_nodes]]]
        lp = self.lp.batch(sources, generator, extra_seeds=task_nodes)

        width = max(c.phenotypes.numel() for c in cases)
        phenotype_index = torch.zeros(len(cases), width, dtype=torch.long)
        phenotype_mask = torch.zeros(len(cases), width, dtype=torch.bool)
        for i, c in enumerate(cases):
            phenotype_index[i, :c.phenotypes.numel()] = lp.seed_index(c.phenotypes)
            phenotype_mask[i, :c.phenotypes.numel()] = True
        positive = torch.stack([torch.isin(candidates, c.truths) for c in cases])
        counts = dict(lp.counts)
        counts.update({"patients": len(cases), "candidates": int(candidates.numel())})
        return TaskBatch(lp, phenotype_index, phenotype_mask, lp.seed_index(candidates),
                         positive, counts)


class DiseaseTask:
    """The disease finetune's step and validation, delegated to by the Trainer."""

    selection_metric = FINETUNE_SELECTION_METRIC

    def __init__(self, *, view: Any, val_cases: Sequence[BoundCase], val_excluded: int,
                 full_edges: Tensor, full_relations: Tensor, task_loss_weight: float,
                 lp_loss_weight: float, ks: Sequence[int] = (1, 3, 5, 10),
                 chunk_size: int = 65536):
        self.view = view
        self.val_cases = list(val_cases)
        self.val_excluded = int(val_excluded)
        if not self.val_cases:
            raise ValueError("no bound validation case; the disease task cannot be validated "
                             "and selected")
        self.full_edges = full_edges
        self.full_relations = full_relations
        self.task_loss_weight = float(task_loss_weight)
        self.lp_loss_weight = float(lp_loss_weight)
        self.ks = tuple(ks)
        self.chunk_size = int(chunk_size)
        self.diseases = view.type_nodes("disease")

    def training_step(self, model: Any, batch: TaskBatch, device: torch.device) -> StepResult:
        lp = batch.lp.to(device)
        z = model.node_vectors_sampled(lp.node_ids, lp.edge_index, lp.nodes_per_hop,
                                       lp.edges_per_hop, edge_type=lp.edge_type)
        head = model.task_head
        patients = patient_vectors(model, z, batch.phenotype_index.to(device),
                                   batch.phenotype_mask.to(device))
        candidates = head.project_candidates(z[batch.candidate_index.to(device)])
        task = nca_loss(patients, candidates, batch.positive.to(device),
                        scale=head.spec.softmax_scale, eps=head.spec.distance_normalisation_eps)
        loss = self.task_loss_weight * task.value
        terms = {"task": float(task.value.detach())}
        counts = dict(batch.counts)
        ok = lp.negative >= 0
        counts["lp_pairs"] = int(ok.sum())
        if bool(ok.any()):
            pos = model.lp_head.score(z[lp.head[ok]], lp.relation[ok], z[lp.tail[ok]])
            neg = model.lp_head.score(z[lp.head[ok]], lp.relation[ok], z[lp.tail[lp.negative[ok]]])
            lp_value = model.lp_head.hinge(pos, neg)
            loss = loss + self.lp_loss_weight * lp_value
            terms["lp"] = float(lp_value.detach())
        else:
            counts["lp_absent"] = 1
        return StepResult(loss, terms, counts)

    def rank_cases(self, model: Any, cases: Sequence[BoundCase], device: torch.device
                   ) -> List[QueryOutcome]:
        """Every case against every disease of the view (U6, E3), by the
        full-view rule (§2.8, G4), through the shared scorer."""
        z = full_view_vectors(model, self.full_edges, self.full_relations, device,
                              self.chunk_size)
        return rank_cases(model, z, cases, self.diseases)

    def validate(self, model: Any, device: torch.device) -> Dict[str, Any]:
        outcomes = self.rank_cases(model, self.val_cases, device)
        outcomes += [QueryOutcome(f"excluded-{i}", INPUT_EXCLUDED, n_positives=1)
                     for i in range(self.val_excluded)]
        report = ranking_report(outcomes, self.ks)
        flat = {k: v for k, v in report.items() if not isinstance(v, dict)}
        flat.update({f"units_{k}": v for k, v in report["units_by_status"].items() if v})
        return flat


__all__ = [
    "Binding",
    "BoundCase",
    "DiseaseTask",
    "FINETUNE_SELECTION_METRIC",
    "FinetuneSampler",
    "LP_SOURCE_TYPES",
    "TaskBatch",
    "bind_disease_cases",
]
