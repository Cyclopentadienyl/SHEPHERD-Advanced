"""
The input module and the GraphEncoder contract, with the reference GATv2.
========================================================================
`docs/working/scorer-retraining/README.md` §2.8, §3.3, C1, C2, U1–U4.

**Two forward rules, one set of modules** (README §2.8):

- `forward_sampled` runs a sampled subgraph whose nodes are ordered by hop
  (seeds first), the layout PyG's neighbour sampler returns. Layer `i` computes
  only the nodes the layers after it still need, from their sampled
  neighbours, as the fixed source's bipartite flow does
  (`up:node_embedder_model.py:121-143`). The graph-mode norm therefore takes
  its statistics over the same node set the source's did.
- `forward_full` runs the whole graph layer by layer in destination chunks.
  Each chunk keeps every incoming edge of its destinations (so each attention
  softmax has its full denominator) and is never an induced subgraph; each
  layer finishes over all nodes before its norm, whose statistics are taken
  over all of them. Dropout is off and BatchNorm uses its running statistics
  in evaluation mode, as anywhere else.

Both call the same `GATv2Conv`, `LayerNorm(mode="graph")`, `BatchNorm1d` and
activation; there is no inference-only copy of the math.

Module: src/models/bundle/encoders.py
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.bundle.spec import EncoderSpec, InputSpec, SpecError


class InputModule(nn.Module):
    """A learned table, one row per view node (README U2), optionally plus a
    node-type embedding (README §3.3 S2; off in the reference)."""

    def __init__(self, spec: InputSpec, num_nodes: int, num_node_types: int):
        super().__init__()
        if num_nodes <= 0:
            raise SpecError("the input module needs at least one node")
        self.spec = spec
        # nn.Embedding initialises N(0, 1), as the source's table
        # (`up:node_embedder_model.py:62`, never re-initialised there).
        self.table = nn.Embedding(num_nodes, spec.dim)
        self.type_embedding: Optional[nn.Embedding] = (
            nn.Embedding(num_node_types, spec.dim) if spec.type_embedding else None)

    def forward(self, node_ids: Tensor, node_types: Optional[Tensor] = None) -> Tensor:
        x = self.table(node_ids)
        if self.type_embedding is not None:
            if node_types is None:
                raise ValueError("this input module adds a type embedding and needs node types")
            x = x + self.type_embedding(node_types)
        return x


class GraphEncoder(nn.Module):
    """The contract every encoder keeps (README §3.3).

    In: input vectors for the nodes of a sampled subgraph or of the full graph,
    message-passing edges (kept apart from supervision edges by the caller),
    and, for encoders that use them, node types and relation ids. Out: one
    vector of `width` per node the caller asked for.
    """

    #: Output width (`EncoderSpec.width`).
    width: int
    uses_node_types: bool = False
    uses_relations: bool = False

    def forward_sampled(self, x: Tensor, edge_index: Tensor, nodes_per_hop: Sequence[int],
                        edges_per_hop: Sequence[int], *, edge_type: Optional[Tensor] = None,
                        node_type: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    def forward_full(self, x: Tensor, edge_index: Tensor, *, edge_type: Optional[Tensor] = None,
                     node_type: Optional[Tensor] = None, chunk_size: int = 65536) -> Tensor:
        raise NotImplementedError


class GATv2Encoder(GraphEncoder):
    """The reference encoder: homogeneous GATv2, heads concatenated, norm and
    activation after every layer but the last (README C1, C2, U1, U4)."""

    def __init__(self, spec: EncoderSpec, in_dim: int, num_relations: int = 0):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, LayerNorm

        if spec.kind != "gatv2":
            raise SpecError(f"GATv2Encoder built for encoder kind {spec.kind!r}")
        self.spec = spec
        self.width = spec.width
        widths: List[int] = []
        for i in range(spec.layers):
            if i == spec.layers - 1:
                widths.append(spec.output)
            elif i == 0:
                widths.append(spec.hidden * spec.first_hidden_multiplier)
            else:
                widths.append(spec.hidden)
        self.uses_relations = spec.relation_edge_dim is not None
        self.relation_embedding: Optional[nn.Embedding] = None
        if self.uses_relations:
            if num_relations <= 0:
                raise SpecError("a relation-aware GATv2 needs the view's relations")
            self.relation_embedding = nn.Embedding(num_relations, spec.relation_edge_dim)
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        dim = in_dim
        for i, out in enumerate(widths):
            self.convs.append(GATv2Conv(
                dim, out, heads=spec.heads, concat=True,
                negative_slope=spec.attention_slope, dropout=spec.attention_dropout,
                add_self_loops=spec.add_self_loops, share_weights=spec.share_weights,
                residual=spec.residual, edge_dim=spec.relation_edge_dim,
                # Explicit zero self-loop edge features (README §3.3 S3); no
                # effect without edge_dim.
                fill_value=0.0 if self.uses_relations else "mean"))
            dim = out * spec.heads
            if i < spec.layers - 1:
                self.layer_norms.append(LayerNorm(dim, mode="graph"))
                self.batch_norms.append(nn.BatchNorm1d(dim))

    def _edge_attr(self, edge_type: Optional[Tensor]) -> Optional[Tensor]:
        if self.relation_embedding is None:
            return None
        if edge_type is None:
            raise ValueError("a relation-aware GATv2 needs edge types")
        return self.relation_embedding(edge_type)

    def _after(self, i: int, x: Tensor) -> Tensor:
        if i == len(self.convs) - 1:
            return x
        x = self.layer_norms[i](x)
        x = F.leaky_relu(x, negative_slope=self.spec.activation_slope)
        x = self.batch_norms[i](x)
        return F.dropout(x, p=self.spec.dropout, training=self.training)

    def forward_sampled(self, x: Tensor, edge_index: Tensor, nodes_per_hop: Sequence[int],
                        edges_per_hop: Sequence[int], *, edge_type: Optional[Tensor] = None,
                        node_type: Optional[Tensor] = None) -> Tensor:
        L = len(self.convs)
        if len(nodes_per_hop) != L + 1 or len(edges_per_hop) != L:
            raise ValueError(f"a {L}-layer encoder needs {L} hops of sampling "
                             f"(got {len(nodes_per_hop) - 1})")
        cum_nodes = torch.tensor([0, *nodes_per_hop]).cumsum(0).tolist()
        cum_edges = torch.tensor([0, *edges_per_hop]).cumsum(0).tolist()
        edge_attr = self._edge_attr(edge_type)
        for i, conv in enumerate(self.convs):
            # Destinations: nodes within L-1-i hops; their sampled incoming
            # edges are the first L-i hops' edges.
            n_dst = cum_nodes[L - i]
            n_src = cum_nodes[L - i + 1]
            e = cum_edges[L - i]
            ei = edge_index[:, :e]
            ea = edge_attr[:e] if edge_attr is not None else None
            x = conv((x[:n_src], x[:n_dst]), ei, edge_attr=ea)
            x = self._after(i, x)
        return x[:cum_nodes[1]]

    @torch.no_grad()
    def forward_full(self, x: Tensor, edge_index: Tensor, *, edge_type: Optional[Tensor] = None,
                     node_type: Optional[Tensor] = None, chunk_size: int = 65536) -> Tensor:
        N = x.size(0)
        edge_attr = self._edge_attr(edge_type)
        dst_all = edge_index[1]
        order = torch.argsort(dst_all, stable=True)
        src_sorted, dst_sorted = edge_index[0][order], dst_all[order]
        ea_sorted = edge_attr[order] if edge_attr is not None else None
        bounds = torch.searchsorted(dst_sorted, torch.arange(0, N + chunk_size, chunk_size,
                                                             device=dst_sorted.device))
        for i, conv in enumerate(self.convs):
            outs = []
            for c, start in enumerate(range(0, N, chunk_size)):
                stop = min(start + chunk_size, N)
                lo, hi = int(bounds[c]), int(bounds[c + 1]) if c + 1 < bounds.numel() else dst_sorted.numel()
                s, d = src_sorted[lo:hi], dst_sorted[lo:hi]
                dst_nodes = torch.arange(start, stop, device=x.device)
                # Sources: the chunk's destinations first (so the layer's own
                # self-loops pair each destination with itself), then every
                # other node with an edge into the chunk.
                other = torch.unique(s)
                other = other[(other < start) | (other >= stop)]
                local_src = torch.cat([dst_nodes, other])
                index = torch.full((N,), -1, dtype=torch.long, device=x.device)
                index[local_src] = torch.arange(local_src.numel(), device=x.device)
                ei = torch.stack([index[s], d - start])
                ea = ea_sorted[lo:hi] if ea_sorted is not None else None
                outs.append(conv((x[local_src], x[dst_nodes]), ei, edge_attr=ea))
            x = torch.cat(outs) if outs else x.new_zeros((0, conv.out_channels * conv.heads))
            x = self._after(i, x)
        return x


__all__ = ["GATv2Encoder", "GraphEncoder", "InputModule"]
