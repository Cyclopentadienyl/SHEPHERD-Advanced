"""
The finetune stage: a task head trained jointly with link prediction.
====================================================================
`docs/working/scorer-retraining/README.md` A6, A7, A10, C3, F5, F7, U6–U9,
G4, E1–E5. The disease task here; the gene and patients-like-me tasks add
their candidate rules to the same sampler and step.

**Binding (F7)** and **ranking (U6)** are the shared task scorer's
(`src.models.tasks.scoring`, and `src.models.tasks.gene` for the gene task),
which measurement and serving use too.

**The gene task (README A8, U9, U14, §6, F9, F10).** Each patient's own
candidate list, plus `upsample_cand` rarely seen genes per patient in training
(U14: one counter over the view's genes, each starting at 1, in the main
process; the batch's candidates are counted, each patient gets its share of
the least-seen genes, ties by view id, minus its causal and listed genes, and
the added genes are counted). The counter is reset at each epoch start and
checkpoints are written at epoch ends, so a resumed epoch draws what an
uninterrupted one would. **In training each patient is scored against the
union of the batch's lists** (upsampled genes included), its own causal genes
the positives, as the source does with its default `only_hard_distractors`
False (`up:gene_prioritization_model.py:127-140`, `up:task_heads/gp_aligner.py:
119-127`). The score is the mixed SPLSIM/EMBSIM with Eq 10 augmentation; the
task term is the multi-similarity loss on it, every causal gene a positive. A
negative with an undefined distance leaves the patient's candidates, a patient
whose causal gene's distance is undefined leaves the step's task term, both
counted (§6 rule 7). Validation ranks each validation case's own list,
without upsampling.

**Patients-like-me (README A9, F9, F11, E1, E5).** Only a patient with at least
two other patients sharing a causal gene anchors (F11); each anchor brings two
of them, drawn seeded, never itself; a group too small to anchor is counted.
The task term is the NCA over the batch's patient vectors, the self excluded,
a pair positive when the causal-gene sets intersect. Validation ranks each
validation patient against every other validation patient (E1's recorded
deviation from the source's in-batch ranking), self excluded by case id.

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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.models.bundle.metrics import QueryOutcome, ranking_report
from src.models.tasks.gene import (
    BoundGeneCase,
    DistanceTable,
    NeighbourTable,
    case_sp,
    gene_list_scores,
    neighbour_inputs,
    rank_gene_cases,
)
from src.models.tasks.losses import multi_similarity_loss, nca_loss
from src.models.tasks.patients_like_me import (
    BoundPatient,
    anchor_groups,
    positive_matrix,
    rank_patients,
)
from src.models.tasks.scoring import (
    Binding,
    BoundCase,
    bind_disease_cases,
    full_view_vectors,
    padded,
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


class _PatientBatches:
    """What the tasks' samplers share (U9): the bound training cases in a
    seeded order per epoch, `batch_size` a batch, and each batch's LP draws over
    the training edges with `sparse_sample_size` random gene, phenotype and
    disease nodes among the seeds."""

    def __init__(self, view: Any, cases: Sequence[Any], lp: BaseLPSampler, *,
                 batch_size: int, sparse_sample_size: int, seed: int):
        if not cases:
            raise ValueError("no bound training case; the finetune has nothing to train on")
        self.view = view
        self.cases = list(cases)
        self.lp = lp
        self.batch_size = int(batch_size)
        self.sparse_sample_size = int(sparse_sample_size)
        self.seed = int(seed)
        self.epoch = 0
        self.sparse_pool = torch.cat([view.type_nodes(t) for t in LP_SOURCE_TYPES])
        types = [view.node_types.index(t) for t in LP_SOURCE_TYPES if t in view.node_types]
        self._lp_type = torch.zeros(len(view.node_types), dtype=torch.bool)
        self._lp_type[types] = True

    def __len__(self) -> int:
        return (len(self.cases) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def start_epoch(self) -> None:
        """Reset what a task keeps across one epoch's batches."""

    def __iter__(self) -> Iterator[Any]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        self.start_epoch()
        order = torch.randperm(len(self.cases), generator=generator).tolist()
        for start in range(0, len(order), self.batch_size):
            yield self.batch([self.cases[i] for i in order[start:start + self.batch_size]],
                             generator)

    def batch(self, cases: Sequence[Any], generator: torch.Generator) -> Any:
        raise NotImplementedError

    def _lp(self, task_nodes: Tensor, generator: torch.Generator) -> LPBatch:
        """The step's subgraph: the task nodes and the sparse sample as seeds,
        those of the LP source types as LP sources (U7)."""
        s = min(self.sparse_sample_size, self.sparse_pool.numel())
        sparse = self.sparse_pool[torch.randperm(self.sparse_pool.numel(), generator=generator)[:s]]
        nodes = torch.unique(torch.cat([task_nodes, sparse]))
        sources = nodes[self._lp_type[self.view.node_type[nodes]]]
        return self.lp.batch(sources, generator, extra_seeds=nodes)


class FinetuneSampler(_PatientBatches):
    """Patient batches for the disease task (U9), reusing the base sampler's
    LP draws over the training edges."""

    def __init__(self, view: Any, cases: Sequence[BoundCase], lp: BaseLPSampler, *,
                 batch_size: int, candidate_sample_size: int, sparse_sample_size: int, seed: int):
        super().__init__(view, cases, lp, batch_size=batch_size,
                         sparse_sample_size=sparse_sample_size, seed=seed)
        self.candidate_sample_size = int(candidate_sample_size)
        self.diseases = view.type_nodes("disease")

    def batch(self, cases: Sequence[BoundCase], generator: torch.Generator) -> TaskBatch:
        truths = torch.unique(torch.cat([c.truths for c in cases]))
        k = min(self.candidate_sample_size, self.diseases.numel())
        sampled = self.diseases[torch.randperm(self.diseases.numel(), generator=generator)[:k]]
        candidates = torch.unique(torch.cat([sampled, truths]))
        phenotypes = torch.cat([c.phenotypes for c in cases])
        lp = self._lp(torch.unique(torch.cat([phenotypes, candidates])), generator)

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


def _lp_term(model: Any, z: Tensor, lp: LPBatch, counts: Dict[str, int]) -> Optional[Tensor]:
    """The step's LP hinge over its pairs with a negative, or None (counted)."""
    ok = lp.negative >= 0
    counts["lp_pairs"] = int(ok.sum())
    if not bool(ok.any()):
        counts["lp_absent"] = 1
        return None
    pos = model.lp_head.score(z[lp.head[ok]], lp.relation[ok], z[lp.tail[ok]])
    neg = model.lp_head.score(z[lp.head[ok]], lp.relation[ok], z[lp.tail[lp.negative[ok]]])
    return model.lp_head.hinge(pos, neg)


def _report(outcomes: List[QueryOutcome], excluded: Sequence[QueryOutcome],
            ks: Sequence[int]) -> Dict[str, Any]:
    """The validation report over the ranked cases and the excluded ones
    (`Binding.excluded_outcomes`), which stay in its denominators (E5)."""
    report = ranking_report(list(outcomes) + list(excluded), ks)
    flat = {k: v for k, v in report.items() if not isinstance(v, dict)}
    flat.update({f"units_{k}": v for k, v in report["units_by_status"].items() if v})
    return flat


class DiseaseTask:
    """The disease finetune's step and validation, delegated to by the Trainer."""

    selection_metric = FINETUNE_SELECTION_METRIC

    def __init__(self, *, view: Any, val_cases: Sequence[BoundCase], val_excluded: Sequence[QueryOutcome],
                 full_edges: Tensor, full_relations: Tensor, task_loss_weight: float,
                 lp_loss_weight: float, ks: Sequence[int] = (1, 3, 5, 10),
                 chunk_size: int = 65536):
        self.view = view
        self.val_cases = list(val_cases)
        self.val_excluded = list(val_excluded)
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
        lp_value = _lp_term(model, z, lp, counts)
        if lp_value is not None:
            loss = loss + self.lp_loss_weight * lp_value
            terms["lp"] = float(lp_value.detach())
        return StepResult(loss, terms, counts)

    def rank_cases(self, model: Any, cases: Sequence[BoundCase], device: torch.device
                   ) -> List[QueryOutcome]:
        """Every case against every disease of the view (U6, E3), by the
        full-view rule (§2.8, G4), through the shared scorer."""
        z = full_view_vectors(model, self.full_edges, self.full_relations, device,
                              self.chunk_size)
        return rank_cases(model, z, cases, self.diseases)

    def validate(self, model: Any, device: torch.device) -> Dict[str, Any]:
        return _report(self.rank_cases(model, self.val_cases, device), self.val_excluded, self.ks)


@dataclass
class GeneBatch:
    """One gene finetune step: the LP subgraph and each patient's own list."""

    lp: LPBatch
    #: `[B, P]` seed indices of the phenotypes, their view ids and the mask.
    phenotype_index: Tensor
    phenotype_view: Tensor
    phenotype_mask: Tensor
    #: `[B, M]` seed indices of the candidates (upsampled included), view ids, mask.
    gene_index: Tensor
    gene_view: Tensor
    gene_mask: Tensor
    #: `[B, M]` causal membership.
    positive: Tensor
    #: `[B, M, K]` seed indices and weights of the augmentation neighbours, `[B, M]` degrees.
    neighbour_index: Tensor
    neighbour_weight: Tensor
    degree: Tensor
    counts: Dict[str, int] = field(default_factory=dict)


class GeneSampler(_PatientBatches):
    """Patient batches for the gene task (U9, U14, F10)."""

    def __init__(self, view: Any, cases: Sequence[BoundGeneCase], lp: BaseLPSampler,
                 neighbours: Optional[NeighbourTable], *, batch_size: int, upsample_size: int,
                 sparse_sample_size: int, seed: int, augmentation_k: int):
        super().__init__(view, cases, lp, batch_size=batch_size,
                         sparse_sample_size=sparse_sample_size, seed=seed)
        self.neighbours = neighbours
        self.upsample_size = int(upsample_size)
        self.k = int(augmentation_k)
        self.genes = view.type_nodes("gene")
        self.gene_pos = torch.full((view.num_nodes,), -1, dtype=torch.long)
        self.gene_pos[self.genes] = torch.arange(self.genes.numel())
        self.counter = torch.ones(self.genes.numel())

    def start_epoch(self) -> None:
        self.counter = torch.ones(self.genes.numel())

    def _upsample(self, cases: Sequence[BoundGeneCase]) -> Tuple[List[Tensor], int]:
        """U14: each patient's list with its share of the least-seen genes."""
        lists = [c.candidates for c in cases]
        u = self.upsample_size
        if u <= 0 or self.genes.numel() == 0:
            return lists, 0
        ones = torch.ones(1)
        listed = self.gene_pos[torch.cat(lists)]
        self.counter.index_add_(0, listed, ones.expand(listed.numel()))
        lowest = self.genes[torch.sort(self.counter, stable=True).indices[:len(cases) * u]]
        out, added = [], []
        for i, case in enumerate(cases):
            share = lowest[i * u:(i + 1) * u]
            share = share[~torch.isin(share, case.truths) & ~torch.isin(share, case.candidates)]
            out.append(torch.cat([case.candidates, share]))
            added.append(share)
        added_pos = self.gene_pos[torch.cat(added)]
        self.counter.index_add_(0, added_pos, ones.expand(added_pos.numel()))
        return out, int(added_pos.numel())

    def batch(self, cases: Sequence[BoundGeneCase], generator: torch.Generator) -> GeneBatch:
        lists, upsampled = self._upsample(cases)
        # Every patient against the union of the batch's lists (the source's
        # training candidates), each row the same union.
        union = torch.unique(torch.cat(lists))
        gene_view = union.unsqueeze(0).expand(len(cases), -1).clone()
        gene_mask = torch.ones_like(gene_view, dtype=torch.bool)
        positive = torch.stack([torch.isin(union, c.truths) for c in cases])
        if self.neighbours is not None:
            near, weight, degree = neighbour_inputs(self.neighbours, gene_view)
            weight = weight * gene_mask.unsqueeze(-1)
        else:
            near = torch.full((*gene_view.shape, self.k), -1, dtype=torch.long)
            weight, degree = torch.zeros(*gene_view.shape, self.k), torch.zeros(gene_view.shape)
        real_near = weight > 0
        phenotype_view, phenotype_mask = padded(cases)
        diseases = torch.tensor([c.disease for c in cases if c.disease >= 0], dtype=torch.long)
        task_nodes = torch.unique(torch.cat([
            phenotype_view[phenotype_mask], gene_view[gene_mask],
            torch.cat([c.truths for c in cases]), diseases, near[real_near]]))
        lp = self._lp(task_nodes, generator)

        def seeds(view_ids: Tensor, mask: Tensor) -> Tensor:
            index = torch.zeros_like(view_ids)
            index[mask] = lp.seed_index(view_ids[mask])
            return index

        counts = dict(lp.counts)
        counts.update({"patients": len(cases), "candidates": int(gene_mask.sum()),
                       "upsampled": upsampled, "genes_with_neighbours":
                       int((real_near.any(-1) & gene_mask).sum())})
        return GeneBatch(lp, seeds(phenotype_view, phenotype_mask), phenotype_view,
                         phenotype_mask, seeds(gene_view, gene_mask), gene_view, gene_mask,
                         positive, seeds(near, real_near), weight, degree, counts)


class GeneTask:
    """The gene finetune's step and validation, delegated to by the Trainer."""

    selection_metric = FINETUNE_SELECTION_METRIC

    def __init__(self, *, view: Any, val_cases: Sequence[BoundGeneCase], val_excluded: Sequence[QueryOutcome],
                 full_edges: Tensor, full_relations: Tensor, distances: DistanceTable,
                 neighbours: Optional[NeighbourTable], task_loss_weight: float,
                 lp_loss_weight: float, ks: Sequence[int] = (1, 3, 5, 10),
                 chunk_size: int = 65536):
        self.view = view
        self.val_cases = list(val_cases)
        self.val_excluded = list(val_excluded)
        if not self.val_cases:
            raise ValueError("no bound validation case; the gene task cannot be validated "
                             "and selected")
        self.full_edges = full_edges
        self.full_relations = full_relations
        self.distances = distances
        self.neighbours = neighbours
        self.task_loss_weight = float(task_loss_weight)
        self.lp_loss_weight = float(lp_loss_weight)
        self.ks = tuple(ks)
        self.chunk_size = int(chunk_size)

    def training_step(self, model: Any, batch: GeneBatch, device: torch.device) -> StepResult:
        lp = batch.lp.to(device)
        z = model.node_vectors_sampled(lp.node_ids, lp.edge_index, lp.nodes_per_hop,
                                       lp.edges_per_hop, edge_type=lp.edge_type)
        spec = model.task_head.spec
        sp, defined = case_sp(self.distances, batch.phenotype_view, batch.phenotype_mask,
                              batch.gene_view, batch.gene_mask, spec.gene)
        scores = gene_list_scores(
            model, z, batch.phenotype_index.to(device), batch.phenotype_mask.to(device),
            batch.gene_index.to(device), batch.neighbour_index.to(device),
            batch.neighbour_weight.to(device), batch.degree.to(device), sp.to(device))
        counts = dict(batch.counts)
        causal_undefined = (batch.positive & ~defined).any(dim=1)
        valid = batch.gene_mask & defined & ~causal_undefined.unsqueeze(1)
        counts["negatives_distance_undefined"] = int(
            (batch.gene_mask & ~batch.positive & ~defined).sum())
        counts["patients_causal_distance_undefined"] = int(causal_undefined.sum())
        alpha, beta, base, margin = spec.multi_similarity
        task = multi_similarity_loss(scores, batch.positive.to(device), valid.to(device),
                                     alpha=alpha, beta=beta, base=base, margin=margin)
        counts["task_anchors"] = task.units
        counts.update({f"task_skipped_{k}": v for k, v in task.skipped.items()})
        loss: Optional[Tensor] = None
        terms: Dict[str, float] = {}
        if task.value is not None:
            loss = self.task_loss_weight * task.value
            terms["task"] = float(task.value.detach())
        else:
            counts["task_absent"] = 1
        lp_value = _lp_term(model, z, lp, counts)
        if lp_value is not None:
            loss = self.lp_loss_weight * lp_value if loss is None \
                else loss + self.lp_loss_weight * lp_value
            terms["lp"] = float(lp_value.detach())
        return StepResult(loss, terms, counts)

    def rank_cases(self, model: Any, cases: Sequence[BoundGeneCase], device: torch.device
                   ) -> List[QueryOutcome]:
        """Each case's own list (no upsampling) by the full-view rule (§2.8, G4)."""
        z = full_view_vectors(model, self.full_edges, self.full_relations, device,
                              self.chunk_size)
        return rank_gene_cases(model, z, cases, self.distances, self.neighbours)

    def validate(self, model: Any, device: torch.device) -> Dict[str, Any]:
        return _report(self.rank_cases(model, self.val_cases, device), self.val_excluded, self.ks)


@dataclass
class PatientBatch:
    """One patients-like-me step: the LP subgraph and the batch's patients."""

    lp: LPBatch
    #: `[N, P]` seed indices of each patient's phenotypes and the mask.
    phenotype_index: Tensor
    phenotype_mask: Tensor
    #: `[N, N]` pairs sharing a causal gene (the diagonal False), and the
    #: patients that are queries of the loss (with a positive in the batch).
    positive: Tensor
    query: Tensor
    counts: Dict[str, int] = field(default_factory=dict)


class PatientsSampler(_PatientBatches):
    """Patients-like-me batches (U9, F11): anchors with two same-gene partners."""

    def __init__(self, view: Any, cases: Sequence[BoundPatient], lp: BaseLPSampler, *,
                 batch_size: int, sparse_sample_size: int, seed: int):
        self.patients = list(cases)
        anchors, self.partners = anchor_groups(self.patients)
        self.index = {c.case_id: i for i, c in enumerate(self.patients)}
        self.non_anchors = len(self.patients) - len(anchors)
        if not anchors:
            raise ValueError("no training patient has two others sharing a causal gene (F11); "
                             "patients-like-me has nothing to anchor on")
        super().__init__(view, [self.patients[i] for i in anchors], lp, batch_size=batch_size,
                         sparse_sample_size=sparse_sample_size, seed=seed)

    def batch(self, anchors: Sequence[BoundPatient], generator: torch.Generator) -> PatientBatch:
        members: List[int] = []
        seen = set()
        for anchor in anchors:
            i = self.index[anchor.case_id]
            if i not in seen:
                members.append(i)
                seen.add(i)
            pool = self.partners[i]
            for pick in torch.randperm(len(pool), generator=generator)[:2].tolist():
                j = pool[pick]
                if j not in seen:
                    members.append(j)
                    seen.add(j)
        patients = [self.patients[i] for i in members]
        positive = positive_matrix(patients, patients)
        phenotype_view, phenotype_mask = padded(patients)
        lp = self._lp(torch.unique(phenotype_view[phenotype_mask]), generator)
        phenotype_index = torch.zeros_like(phenotype_view)
        phenotype_index[phenotype_mask] = lp.seed_index(phenotype_view[phenotype_mask])
        query = positive.any(dim=1)
        counts = dict(lp.counts)
        counts.update({"anchors": len(anchors), "patients": len(patients),
                       "queries": int(query.sum()), "non_anchoring_patients": self.non_anchors})
        return PatientBatch(lp, phenotype_index, phenotype_mask, positive, query, counts)


class PatientsTask:
    """The patients-like-me finetune's step and validation."""

    selection_metric = FINETUNE_SELECTION_METRIC

    def __init__(self, *, view: Any, val_cases: Sequence[BoundPatient], val_excluded: Sequence[QueryOutcome],
                 full_edges: Tensor, full_relations: Tensor, task_loss_weight: float,
                 lp_loss_weight: float, ks: Sequence[int] = (1, 3, 5, 10),
                 chunk_size: int = 65536):
        self.view = view
        self.val_cases = list(val_cases)
        self.val_excluded = list(val_excluded)
        if not bool(positive_matrix(self.val_cases, self.val_cases).any()):
            raise ValueError("no validation patient shares a causal gene with another; "
                             "patients-like-me cannot be validated and selected")
        self.full_edges = full_edges
        self.full_relations = full_relations
        self.task_loss_weight = float(task_loss_weight)
        self.lp_loss_weight = float(lp_loss_weight)
        self.ks = tuple(ks)
        self.chunk_size = int(chunk_size)

    def training_step(self, model: Any, batch: PatientBatch, device: torch.device) -> StepResult:
        lp = batch.lp.to(device)
        z = model.node_vectors_sampled(lp.node_ids, lp.edge_index, lp.nodes_per_hop,
                                       lp.edges_per_hop, edge_type=lp.edge_type)
        spec = model.task_head.spec
        patients = patient_vectors(model, z, batch.phenotype_index.to(device),
                                   batch.phenotype_mask.to(device))
        query = batch.query.to(device)
        n = patients.size(0)
        task = nca_loss(patients[query], patients, batch.positive.to(device)[query],
                        scale=spec.softmax_scale, eps=spec.distance_normalisation_eps,
                        exclude=torch.eye(n, dtype=torch.bool, device=device)[query])
        loss = self.task_loss_weight * task.value
        terms = {"task": float(task.value.detach())}
        counts = dict(batch.counts)
        lp_value = _lp_term(model, z, lp, counts)
        if lp_value is not None:
            loss = loss + self.lp_loss_weight * lp_value
            terms["lp"] = float(lp_value.detach())
        return StepResult(loss, terms, counts)

    def rank_cases(self, model: Any, cases: Sequence[BoundPatient], device: torch.device
                   ) -> List[QueryOutcome]:
        """Each patient against every other one of `cases` (E1), full view (G4)."""
        z = full_view_vectors(model, self.full_edges, self.full_relations, device,
                              self.chunk_size)
        return rank_patients(model, z, cases, cases)

    def validate(self, model: Any, device: torch.device) -> Dict[str, Any]:
        return _report(self.rank_cases(model, self.val_cases, device), self.val_excluded, self.ks)


__all__ = [
    "Binding",
    "BoundCase",
    "DiseaseTask",
    "FINETUNE_SELECTION_METRIC",
    "FinetuneSampler",
    "GeneBatch",
    "GeneSampler",
    "GeneTask",
    "LP_SOURCE_TYPES",
    "PatientBatch",
    "PatientsSampler",
    "PatientsTask",
    "TaskBatch",
    "bind_disease_cases",
]
