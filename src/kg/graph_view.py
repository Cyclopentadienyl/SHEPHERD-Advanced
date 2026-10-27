"""
The reference graph view: the identified graph a model bundle reads.
====================================================================
`docs/working/scorer-retraining/README.md` G1, G4, G5 and F2.

A workspace exports a heterogeneous graph: per-type node counts and one
`(src_type, rel, dst_type)` edge tensor per relation, every relation with a
`rev_` twin (`KnowledgeGraph.export_to_model`). The view turns that into one
homogeneous graph with typed edges, in a fixed order:

1. **Included relations.** Every exported relation except those the spec
   excludes; a relation's reverse twin leaves with it.
2. **Native self-loops dropped** (an edge whose two ends are one node). The
   GATv2 layers' own self-loops and the sampler's artificial self pairs are
   other things and are untouched by this (README F6).
3. **Duplicates dropped by relation semantics:** the same two nodes joined by
   the same relation twice keep one edge; different relations between the same
   nodes stay distinct.
4. **The largest connected component** of that full undirected graph, ties
   broken toward the component holding the smallest node id. Nodes outside it
   are excluded from this view, keep their ids and are counted, so coverage
   denominators never shrink silently.
5. **Only then the split** (`split_facts`): an edge and its reverse twin are one
   fact and always land in the same part.

**Nothing on disk is changed.** The view is computed from the workspace's
graph read and the spec's rule, deterministically, and identified by digest:
the same graph and rule give the same view on any machine.

Global ids: node types in sorted order, each type's local indices offset by the
counts before it. The view's node ids are those global ids that survive, in
ascending order (`view_to_raw`), so every consumer shares one mapping.

Module: src/kg/graph_view.py
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

VIEW_RULE = "reference/v1"
SPLIT_RULE = "facts-80/10/10/v1"
SPLIT_NAMES = ("train", "val", "test")
REVERSE_PREFIX = "rev_"


class GraphViewError(ValueError):
    """A graph the view cannot be formed from, or an inconsistent request."""


def relation_key(edge_type: Sequence[str]) -> str:
    src, rel, dst = edge_type
    return f"{src}|{rel}|{dst}"


def _reverse_key(key: str) -> str:
    src, rel, dst = key.split("|")
    if rel.startswith(REVERSE_PREFIX):
        return f"{dst}|{rel[len(REVERSE_PREFIX):]}|{src}"
    return f"{dst}|{REVERSE_PREFIX}{rel}|{src}"


@dataclass
class GraphView:
    """One homogeneous, typed graph and how it came from the workspace graph."""

    rule: Dict[str, Any]
    node_types: List[str]
    type_offsets: Dict[str, int]
    raw_num_nodes: int
    #: View node id -> global (raw) id, ascending.
    view_to_raw: torch.Tensor
    #: Global id -> view node id, -1 for an excluded node.
    raw_to_view: torch.Tensor
    #: View node id -> index into `node_types`.
    node_type: torch.Tensor
    relations: List[str]
    #: Relation id -> its reverse twin's relation id, -1 when it has none.
    reverse_relation: torch.Tensor
    #: View edges: source, destination (view ids) and relation id.
    src: torch.Tensor
    dst: torch.Tensor
    rel: torch.Tensor
    #: Each edge's fact id: an edge and its reverse twin share one.
    fact: torch.Tensor
    counts: Dict[str, Any] = field(default_factory=dict)
    digest: str = ""

    @property
    def num_nodes(self) -> int:
        return int(self.view_to_raw.numel())

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    @property
    def num_facts(self) -> int:
        return int(self.fact.max().item()) + 1 if self.fact.numel() else 0

    def edge_index(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`[2, E]` (source, destination) of the view's edges, or of the ones
        `mask` selects."""
        if mask is None:
            return torch.stack([self.src, self.dst])
        return torch.stack([self.src[mask], self.dst[mask]])

    def view_ids(self, node_type: str, local: torch.Tensor) -> torch.Tensor:
        """View ids of `node_type`'s local indices; -1 where excluded."""
        if node_type not in self.type_offsets:
            raise GraphViewError(f"node type {node_type!r} is not in the graph")
        raw = torch.as_tensor(local, dtype=torch.long) + self.type_offsets[node_type]
        return self.raw_to_view[raw]

    def type_nodes(self, node_type: str) -> torch.Tensor:
        """The view ids of every node of `node_type`, ascending."""
        if node_type not in self.node_types:
            return torch.empty(0, dtype=torch.long)
        t = self.node_types.index(node_type)
        return torch.nonzero(self.node_type == t, as_tuple=False).flatten()

    def local_index(self, view_ids: torch.Tensor) -> torch.Tensor:
        """The type-local index of each view node (its row in the workspace's
        per-type tensors)."""
        raw = self.view_to_raw[view_ids]
        offsets = torch.tensor([self.type_offsets[t] for t in self.node_types], dtype=torch.long)
        return raw - offsets[self.node_type[view_ids]]

    def record(self) -> Dict[str, Any]:
        """What a checkpoint records about this view (plain data)."""
        return {"rule": dict(self.rule), "digest": self.digest,
                "num_nodes": self.num_nodes, "num_edges": self.num_edges,
                "num_facts": self.num_facts, "node_types": list(self.node_types),
                "relations": list(self.relations), "counts": dict(self.counts)}


def build_graph_view(graph_data: Mapping[str, Any], *, rule: Optional[Mapping[str, Any]] = None
                     ) -> GraphView:
    """The view of a workspace graph read (`read_graph_artifacts`' `graph_data`).

    `rule` is the spec's `graph_view` section as plain data
    (`GraphViewSpec`); omitted, the reference rule.
    """
    rule = dict(rule or {"rule": VIEW_RULE, "drop_native_self_loops": True, "deduplicate": True,
                         "largest_component": True, "exclude_relations": []})
    if rule.get("rule") != VIEW_RULE:
        raise GraphViewError(f"unknown graph view rule {rule.get('rule')!r}")
    num_nodes_dict = graph_data.get("num_nodes_dict")
    edge_index_dict = graph_data.get("edge_index_dict")
    if not num_nodes_dict or edge_index_dict is None:
        raise GraphViewError("the graph read has no num_nodes_dict or edge_index_dict")

    node_types = sorted(num_nodes_dict)
    offsets: Dict[str, int] = {}
    total = 0
    for t in node_types:
        offsets[t] = total
        total += int(num_nodes_dict[t])

    excluded_rel = {str(r) for r in rule.get("exclude_relations", [])}
    excluded_rel |= {_reverse_key(r) for r in excluded_rel}
    present = {relation_key(k) for k in edge_index_dict}
    unknown_excluded = sorted(r for r in rule.get("exclude_relations", []) if r not in present)
    if unknown_excluded:
        raise GraphViewError(f"graph_view.exclude_relations names relations the graph does not "
                             f"have: {unknown_excluded}")

    relations = sorted(r for r in present if r not in excluded_rel)
    rel_index = {r: i for i, r in enumerate(relations)}
    reverse = torch.full((len(relations),), -1, dtype=torch.long)
    for r, i in rel_index.items():
        twin = _reverse_key(r)
        if twin in rel_index:
            reverse[i] = rel_index[twin]

    srcs, dsts, rels = [], [], []
    for key_tuple, ei in edge_index_dict.items():
        key = relation_key(key_tuple)
        if key not in rel_index:
            continue
        s_type, _, d_type = key_tuple
        for t in (s_type, d_type):
            if t not in offsets:
                raise GraphViewError(f"relation {key} names node type {t!r}, which has no count")
        ei = torch.as_tensor(ei, dtype=torch.long)
        if ei.dim() != 2 or ei.size(0) != 2:
            raise GraphViewError(f"relation {key}'s edge tensor is not [2, E]")
        if ei.numel():
            if int(ei[0].max()) >= int(num_nodes_dict[s_type]) or \
                    int(ei[1].max()) >= int(num_nodes_dict[d_type]) or int(ei.min()) < 0:
                raise GraphViewError(f"relation {key} has an index outside its node counts")
        srcs.append(ei[0] + offsets[s_type])
        dsts.append(ei[1] + offsets[d_type])
        rels.append(torch.full((ei.size(1),), rel_index[key], dtype=torch.long))
    src = torch.cat(srcs) if srcs else torch.empty(0, dtype=torch.long)
    dst = torch.cat(dsts) if dsts else torch.empty(0, dtype=torch.long)
    rel = torch.cat(rels) if rels else torch.empty(0, dtype=torch.long)
    counts: Dict[str, Any] = {"edges_exported": int(src.numel())}

    if rule.get("drop_native_self_loops", True):
        keep = src != dst
        counts["native_self_loops_dropped"] = int((~keep).sum())
        src, dst, rel = src[keep], dst[keep], rel[keep]

    if rule.get("deduplicate", True):
        R = max(len(relations), 1)
        keys = (src * total + dst) * R + rel
        unique_keys, first = np.unique(keys.numpy(), return_index=True)
        order = np.sort(first)
        counts["duplicates_dropped"] = int(src.numel() - order.size)
        idx = torch.from_numpy(order)
        src, dst, rel = src[idx], dst[idx], rel[idx]

    keep_nodes = torch.ones(total, dtype=torch.bool)
    if rule.get("largest_component", True) and total:
        keep_nodes = _largest_component(src, dst, total)
        counts["components"] = int(_component_count(src, dst, total))
    excluded_per_type = {}
    for t in node_types:
        lo, hi = offsets[t], offsets[t] + int(num_nodes_dict[t])
        excluded_per_type[t] = int((~keep_nodes[lo:hi]).sum())
    counts["nodes_excluded_by_type"] = excluded_per_type

    edge_keep = keep_nodes[src] & keep_nodes[dst] if src.numel() else torch.empty(0, dtype=torch.bool)
    counts["edges_outside_component"] = int((~edge_keep).sum()) if src.numel() else 0
    src, dst, rel = src[edge_keep], dst[edge_keep], rel[edge_keep]

    view_to_raw = torch.nonzero(keep_nodes, as_tuple=False).flatten()
    raw_to_view = torch.full((total,), -1, dtype=torch.long)
    raw_to_view[view_to_raw] = torch.arange(view_to_raw.numel())
    type_of_raw = torch.empty(total, dtype=torch.long)
    for i, t in enumerate(node_types):
        type_of_raw[offsets[t]:offsets[t] + int(num_nodes_dict[t])] = i
    node_type = type_of_raw[view_to_raw]
    src, dst = raw_to_view[src], raw_to_view[dst]

    fact = _facts(src, dst, rel, reverse, relations, view_to_raw.numel())

    view = GraphView(
        rule={**rule, "exclude_relations": sorted(rule.get("exclude_relations", []))},
        node_types=node_types, type_offsets=offsets, raw_num_nodes=total,
        view_to_raw=view_to_raw, raw_to_view=raw_to_view, node_type=node_type,
        relations=relations, reverse_relation=reverse, src=src, dst=dst, rel=rel, fact=fact,
        counts=counts,
    )
    view.digest = view_digest(view)
    logger.info("Graph view %s: %d of %d nodes, %d edges, %d facts, %d relations; %s",
                view.digest[:12], view.num_nodes, total, view.num_edges, view.num_facts,
                len(relations), counts)
    return view


def _component_labels(src: torch.Tensor, dst: torch.Tensor, n: int) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    data = np.ones(src.numel(), dtype=np.int8)
    adj = coo_matrix((data, (src.numpy(), dst.numpy())), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def _component_count(src: torch.Tensor, dst: torch.Tensor, n: int) -> int:
    return int(np.unique(_component_labels(src, dst, n)).size)


def _largest_component(src: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    labels = _component_labels(src, dst, n)
    sizes = np.bincount(labels)
    best = sizes.max()
    # Ties: the component holding the smallest node id (labels are assigned in
    # node order, so the smallest label among the largest is that component).
    chosen = int(np.flatnonzero(sizes == best).min())
    return torch.from_numpy(labels == chosen)


def _facts(src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor, reverse: torch.Tensor,
           relations: Sequence[str], n: int) -> torch.Tensor:
    """One fact id per edge, shared by an edge and its reverse twin.

    An edge of a relation named with the `rev_` prefix, whose twin relation is
    present, is a reverse edge and joins the fact of the forward edge it
    mirrors; every other edge starts a fact. A reverse edge whose forward edge
    is missing (excluded as a duplicate, say) starts its own fact.
    """
    E = src.numel()
    if E == 0:
        return torch.empty(0, dtype=torch.long)
    R = max(len(relations), 1)
    rev = reverse.numpy()
    reverse_rel = np.array([rev[i] >= 0 and r.split("|")[1].startswith(REVERSE_PREFIX)
                            for i, r in enumerate(relations)], dtype=bool)
    src_np, dst_np, rel_np = src.numpy(), dst.numpy(), rel.numpy()
    keys = (src_np.astype(np.int64) * n + dst_np) * R + rel_np
    is_reverse = reverse_rel[rel_np]
    forward_idx = np.flatnonzero(~is_reverse)
    fact_np = np.full(E, -1, dtype=np.int64)
    fact_np[forward_idx] = np.arange(forward_idx.size)
    order = np.argsort(keys[forward_idx], kind="stable")
    sorted_keys = keys[forward_idx][order]
    rev_idx = np.flatnonzero(is_reverse)
    if rev_idx.size:
        s, d, r = src_np[rev_idx], dst_np[rev_idx], rel_np[rev_idx]
        twin_keys = (d.astype(np.int64) * n + s) * R + rev[r]
        if sorted_keys.size:
            pos = np.searchsorted(sorted_keys, twin_keys)
            safe = np.minimum(pos, sorted_keys.size - 1)
            found = (pos < sorted_keys.size) & (sorted_keys[safe] == twin_keys)
        else:
            pos = np.zeros(rev_idx.size, dtype=np.int64)
            found = np.zeros(rev_idx.size, dtype=bool)
        matched = rev_idx[found]
        fact_np[matched] = fact_np[forward_idx[order[pos[found]]]]
        orphans = rev_idx[~found]
        fact_np[orphans] = forward_idx.size + np.arange(orphans.size)
    # Fact ids in first-seen edge order, so the numbering does not depend on
    # how the arrays above were ordered internally.
    _, first, inverse = np.unique(fact_np, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    return torch.from_numpy(rank[inverse])


def view_digest(view: GraphView) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"rule": view.rule, "node_types": view.node_types,
                         "type_offsets": view.type_offsets, "relations": view.relations},
                        sort_keys=True).encode())
    for t in (view.view_to_raw, view.src, view.dst, view.rel, view.fact):
        h.update(t.to(torch.int64).contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class EdgeSplit:
    """Which part each view edge is in (F2): 0 train, 1 val, 2 test."""

    part: torch.Tensor
    seed: int
    view_digest: str
    rule: str = SPLIT_RULE
    digest: str = ""

    def mask(self, name: str) -> torch.Tensor:
        return self.part == SPLIT_NAMES.index(name)

    def record(self) -> Dict[str, Any]:
        sizes = {n: int(self.mask(n).sum()) for n in SPLIT_NAMES}
        return {"rule": self.rule, "seed": int(self.seed), "view_digest": self.view_digest,
                "digest": self.digest, "edges": sizes}


def split_facts(view: GraphView, seed: int) -> EdgeSplit:
    """80/10/10 over facts, so an edge and its reverse twin share a part.

    NumPy's legacy `RandomState` gives the same permutation for the same seed
    across NumPy versions; the permutation, not a hash of the order, decides.
    """
    n = view.num_facts
    perm = np.random.RandomState(int(seed)).permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    fact_part = np.empty(n, dtype=np.int64)
    fact_part[perm[:n_train]] = 0
    fact_part[perm[n_train:n_train + n_val]] = 1
    fact_part[perm[n_train + n_val:]] = 2
    part = torch.from_numpy(fact_part)[view.fact] if view.num_edges else torch.empty(0, dtype=torch.long)
    split = EdgeSplit(part=part, seed=int(seed), view_digest=view.digest)
    h = hashlib.sha256()
    h.update(json.dumps({"rule": SPLIT_RULE, "seed": int(seed), "view": view.digest},
                        sort_keys=True).encode())
    h.update(part.to(torch.int64).contiguous().numpy().tobytes())
    split.digest = h.hexdigest()
    return split


__all__ = [
    "EdgeSplit",
    "GraphView",
    "GraphViewError",
    "SPLIT_NAMES",
    "SPLIT_RULE",
    "VIEW_RULE",
    "build_graph_view",
    "relation_key",
    "split_facts",
    "view_digest",
]
