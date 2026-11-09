"""
The base stage: link prediction over the graph view (the paper's pretraining).
=============================================================================
`docs/working/scorer-retraining/README.md` A2–A5, C4, U3, U5, F1, F3–F6, G4.

**Sampler (C4, F5, F6).** Each epoch visits every view node once as a source,
in a seeded order, `batch_size` sources a batch. A source draws one step among
its supervision out-edges and one artificial self entry, each with probability
`1/(d+1)` (`d` counts edges, as the source's random walk does over its
adjacency, `up:samplers.py:110-111,203`). A real draw `s → t` makes every
supervision edge from `s` to `t` a positive, one per edge
(`up:utils/pretrain_utils.py:98-115`); a self draw makes none and is counted
(F6), and a source without an edge can only draw itself. The batch's seeds are
the distinct sources and targets; PyG's neighbour sampler takes `sizes` hops
of in-neighbours over the message-passing edges, and then every sampled edge
between a drawn pair, in either direction and of any relation, leaves the
step's message-passing graph (F5, `up:samplers.py:177-193`). The source fed the
concatenated sources and targets, duplicates included, to its sampler; distinct
seeds are used here, so a node repeated in a batch does not weigh twice in the
graph-mode norm.

**Negatives (U5, F1).** One corrupted tail per positive, drawn among the tails of
the same relation's positives in the batch and never the positive's own tail;
a positive without such a tail adds no term and is counted. The sampler draws
them with the batch, from the epoch's seeded stream, so a resumed run draws
what an uninterrupted one would.

**Graphs (A4, G4).** Message passing in training and in validation runs over
the training edges only; supervision comes from the training edges in
training and from the validation edges in validation.

**Validation (F3, F4, A5).** Every validation edge once, in a seeded fixed
order and fixed batches, against negatives drawn once by the same rule under
the recorded seed; node vectors by full-view chunked inference (§2.8) over the
training edges; whole-set LP accuracy (the selection metric), ROC-AUC and AP.
Two evaluations of the same weights give identical numbers.

Module: src/training/lp_task.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from torch import Tensor

from src.models.bundle.lp_head import corrupt_tails
from src.models.bundle.metrics import lp_metrics

#: The selection metric of the base stage (A5), as the Trainer logs it.
BASE_SELECTION_METRIC = "val_lp_accuracy"


@dataclass
class LPBatch:
    """One base training step's sampled subgraph and supervision."""

    #: View node ids of the subgraph, hop-ordered, seeds first.
    node_ids: Tensor
    #: `[2, E]` message-passing edges in local ids (row = source), hop-ordered.
    edge_index: Tensor
    #: The view relation of each message-passing edge.
    edge_type: Tensor
    nodes_per_hop: List[int]
    edges_per_hop: List[int]
    #: Positives: local head (source) index, relation, local tail index.
    head: Tensor
    relation: Tensor
    tail: Tensor
    #: For each positive, the positive whose tail is its corrupted tail, or -1.
    negative: Tensor
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def num_seeds(self) -> int:
        return int(self.nodes_per_hop[0])

    def seed_index(self, view_ids: Tensor) -> Tensor:
        """Local indices of seed nodes (the seeds are the distinct, ascending
        view ids at the front of `node_ids`)."""
        seeds = self.node_ids[:self.num_seeds]
        index = torch.searchsorted(seeds, view_ids.to(seeds.device))
        if bool((index >= seeds.numel()).any()) or not torch.equal(seeds[index], view_ids.to(seeds.device)):
            raise ValueError("a node asked for is not a seed of this batch")
        return index

    def to(self, device: torch.device) -> "LPBatch":
        return LPBatch(self.node_ids.to(device), self.edge_index.to(device),
                       self.edge_type.to(device), self.nodes_per_hop, self.edges_per_hop,
                       self.head.to(device), self.relation.to(device), self.tail.to(device),
                       self.negative.to(device), dict(self.counts))


@dataclass
class StepResult:
    """What a task step gives the Trainer: the loss, or None when the step has
    nothing to learn from (no backward, no update), its named terms and counts."""

    loss: Optional[Tensor]
    terms: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


class _EdgeTable:
    """Edges sorted by (source, target) key, for draws and positive lookup."""

    def __init__(self, src: Tensor, dst: Tensor, rel: Tensor, num_nodes: int):
        self.n = num_nodes
        key = src * num_nodes + dst
        order = torch.argsort(key, stable=True)
        self.key, self.src, self.dst, self.rel = key[order], src[order], dst[order], rel[order]
        self.rowptr = torch.searchsorted(self.src, torch.arange(num_nodes + 1))

    def out_degree(self, nodes: Tensor) -> Tensor:
        return self.rowptr[nodes + 1] - self.rowptr[nodes]

    def edges_between(self, heads: Tensor, tails: Tensor) -> Tensor:
        """Indices (into this table) of every edge `heads[i] → tails[i]`, and
        the pair each belongs to."""
        keys = heads * self.n + tails
        lo = torch.searchsorted(self.key, keys, right=False)
        hi = torch.searchsorted(self.key, keys, right=True)
        counts = hi - lo
        pair = torch.repeat_interleave(torch.arange(keys.numel()), counts)
        offsets = torch.arange(int(counts.sum())) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts)
        return torch.repeat_interleave(lo, counts) + offsets, pair


class BaseLPSampler:
    """Source-visit batches for the base stage (C4, F5, F6)."""

    def __init__(self, num_nodes: int, message_passing: Tensor, message_relations: Tensor,
                 supervision: Tensor, supervision_relations: Tensor, *, sizes: Sequence[int],
                 batch_size: int, seed: int):
        from torch_geometric.data import Data
        from torch_geometric.sampler import NeighborSampler

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_nodes = int(num_nodes)
        self.sizes = [int(s) for s in sizes]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mp_relations = message_relations
        self._sampler = NeighborSampler(
            Data(edge_index=message_passing.contiguous(), num_nodes=self.num_nodes),
            num_neighbors=self.sizes)
        self._sup = _EdgeTable(supervision[0], supervision[1], supervision_relations,
                               self.num_nodes)
        self.epoch = 0

    def __len__(self) -> int:
        return (self.num_nodes + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[LPBatch]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(self.num_nodes, generator=generator)
        for start in range(0, self.num_nodes, self.batch_size):
            yield self.batch(order[start:start + self.batch_size], generator)

    def draw(self, sources: Tensor, generator: torch.Generator) -> Tensor:
        """One target per source: a supervision out-neighbour, or the source
        itself with probability 1/(d+1)."""
        degree = self._sup.out_degree(sources)
        pick = (torch.rand(sources.numel(), generator=generator) * (degree + 1)).long()
        pick = torch.minimum(pick, degree)
        is_self = pick == degree
        edge = (self._sup.rowptr[sources] + pick).clamp(max=max(self._sup.dst.numel() - 1, 0))
        targets = sources.clone()
        if self._sup.dst.numel():
            targets[~is_self] = self._sup.dst[edge[~is_self]]
        return targets

    def batch(self, sources: Tensor, generator: torch.Generator,
              extra_seeds: Optional[Tensor] = None) -> LPBatch:
        """One step's subgraph: the LP draws of `sources`, and `extra_seeds`
        (a finetune's task nodes) among the seeds too."""
        from torch_geometric.sampler import NodeSamplerInput

        targets = self.draw(sources, generator)
        real = targets != sources
        isolated = int((self._sup.out_degree(sources) == 0).sum())
        heads, tails = sources[real], targets[real]
        idx, _ = self._sup.edges_between(heads, tails)
        pos_head, pos_tail, pos_rel = self._sup.src[idx], self._sup.dst[idx], self._sup.rel[idx]

        seed_parts = [sources, targets] + ([extra_seeds] if extra_seeds is not None else [])
        seeds = torch.unique(torch.cat(seed_parts))
        out = self._sampler.sample_from_nodes(NodeSamplerInput(input_id=None, node=seeds))
        if out.num_sampled_nodes is None or out.num_sampled_edges is None:
            # torch-sparse's fallback gives no per-hop counts; the layer-wise
            # sampled forward (§2.8) needs them, so pyg-lib is required.
            raise RuntimeError("the neighbour sampler returned no per-hop node and edge counts; "
                               "the base stage needs pyg-lib's sampler (see the pinned PyG set)")
        node, row, col = out.node, out.row, out.col
        edge = self._sampler.perm[out.edge] if self._sampler.perm is not None else out.edge

        # F5: every sampled edge between a drawn pair, both directions, any
        # relation, leaves this step's message passing.
        n = self.num_nodes
        a, b = node[row], node[col]
        sampled_key = torch.minimum(a, b) * n + torch.maximum(a, b)
        pair_key = torch.minimum(heads, tails) * n + torch.maximum(heads, tails)
        keep = ~torch.isin(sampled_key, pair_key)
        hop_of_edge = torch.repeat_interleave(
            torch.arange(len(out.num_sampled_edges)), torch.tensor(out.num_sampled_edges))
        edges_per_hop = torch.bincount(hop_of_edge[keep],
                                       minlength=len(out.num_sampled_edges)).tolist()

        local = torch.full((n,), -1, dtype=torch.long)
        local[node] = torch.arange(node.numel())
        negative, has_negative = corrupt_tails(pos_rel, pos_tail, generator)
        counts = {
            "source_visits": int(sources.numel()),
            "self_draws": int((~real).sum()),
            "isolated_sources": isolated,
            "drawn_pairs": int(real.sum()),
            "lp_positives": int(idx.numel()),
            "lp_without_negative": int((~has_negative).sum()),
            "edges_filtered": int((~keep).sum()),
        }
        return LPBatch(
            node_ids=node, edge_index=torch.stack([row[keep], col[keep]]),
            edge_type=self.mp_relations[edge[keep]],
            nodes_per_hop=list(out.num_sampled_nodes), edges_per_hop=edges_per_hop,
            head=local[pos_head], relation=pos_rel, tail=local[pos_tail], negative=negative,
            counts=counts)


class BaseLPTask:
    """The base stage's step and validation, delegated to by the Trainer."""

    selection_metric = BASE_SELECTION_METRIC

    def __init__(self, *, num_nodes: int, train_edges: Tensor, train_relations: Tensor,
                 val_edges: Tensor, val_relations: Tensor, seed: int,
                 val_batch_size: int = 512, chunk_size: int = 65536):
        self.num_nodes = int(num_nodes)
        self.train_edges = train_edges
        self.train_relations = train_relations
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.val_head, self.val_rel, self.val_tail, self.val_neg_tail, self.val_counts = \
            self._fixed_validation(val_edges, val_relations, int(val_batch_size))
        if self.val_head.numel() == 0:
            # Refused before the job starts, not at the first validation.
            raise ValueError(
                f"none of the {self.val_counts['val_edges']} validation edges has a negative "
                "(no other tail of its relation in its batch), so the base stage cannot be "
                "validated on this split")

    def _fixed_validation(self, edges: Tensor, relations: Tensor, batch_size: int):
        """Every validation edge once, in a seeded order and fixed batches, with
        one negative drawn once per edge by U5's rule (F3)."""
        generator = torch.Generator().manual_seed(self.seed + 7919)
        order = torch.randperm(edges.size(1), generator=generator)
        heads, tails, rels = edges[0][order], edges[1][order], relations[order]
        keep_h, keep_r, keep_t, neg = [], [], [], []
        without = 0
        for start in range(0, heads.numel(), batch_size):
            h = heads[start:start + batch_size]
            t = tails[start:start + batch_size]
            r = rels[start:start + batch_size]
            chosen, ok = corrupt_tails(r, t, generator)
            without += int((~ok).sum())
            keep_h.append(h[ok])
            keep_r.append(r[ok])
            keep_t.append(t[ok])
            neg.append(t[chosen[ok]])
        cat = (lambda xs: torch.cat(xs) if xs else torch.empty(0, dtype=torch.long))
        counts = {"val_edges": int(heads.numel()), "val_without_negative": without}
        return cat(keep_h), cat(keep_r), cat(keep_t), cat(neg), counts

    def training_step(self, model: Any, batch: LPBatch, device: torch.device) -> StepResult:
        counts = dict(batch.counts)
        pairs = int((batch.negative >= 0).sum())
        counts["lp_pairs"] = pairs
        if pairs == 0:
            return StepResult(None, {}, counts)
        batch = batch.to(device)
        z = model.node_vectors_sampled(batch.node_ids, batch.edge_index, batch.nodes_per_hop,
                                       batch.edges_per_hop, edge_type=batch.edge_type)
        ok = batch.negative >= 0
        head, rel, tail = batch.head[ok], batch.relation[ok], batch.tail[ok]
        neg_tail = batch.tail[batch.negative[ok]]
        positive = model.lp_head.score(z[head], rel, z[tail])
        negative = model.lp_head.score(z[head], rel, z[neg_tail])
        loss = model.lp_head.hinge(positive, negative)
        return StepResult(loss, {"lp": float(loss.detach())}, counts)

    @torch.no_grad()
    def validate(self, model: Any, device: torch.device) -> Dict[str, Any]:
        model.eval()
        z = model.node_vectors_full(self.train_edges.to(device),
                                    edge_type=self.train_relations.to(device),
                                    chunk_size=self.chunk_size)
        h = z[self.val_head.to(device)]
        r = self.val_rel.to(device)
        pos = model.lp_head.raw_score(h, r, z[self.val_tail.to(device)])
        neg = model.lp_head.raw_score(h, r, z[self.val_neg_tail.to(device)])
        metrics = lp_metrics(pos.cpu().tolist(), neg.cpu().tolist())
        hinge = model.lp_head.hinge(torch.tanh(pos), torch.tanh(neg))
        metrics["lp_loss"] = float(hinge)
        metrics.update(self.val_counts)
        return metrics


__all__ = ["BASE_SELECTION_METRIC", "BaseLPSampler", "BaseLPTask", "LPBatch", "StepResult"]
