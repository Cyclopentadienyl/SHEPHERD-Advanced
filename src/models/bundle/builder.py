"""
The one builder: a model spec and a graph view's shape → one `nn.Module`.
========================================================================
`docs/working/scorer-retraining/README.md` §3.1, §4.

Every trainable parameter (node table, encoder, relation vectors, patient
encoder, task head) is a submodule of the returned `ModelBundle`. The builder
reads the spec and the view's shape only: no weight name, state-dict shape or
training flag decides what is built. The Trainer's optimizer, the checkpoint
writer and reader, serving, measurement and the scripts all build through here.

Module: src/models/bundle/builder.py
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bundle.encoders import GATv2Encoder, GraphEncoder, InputModule
from src.models.bundle.lp_head import DistMultHead
from src.models.bundle.spec import ModelSpec, SpecError, ViewShape
from src.models.tasks.heads import PatientEncoder, TaskHead, build_task_head


class ModelBundle(nn.Module):
    """Input module → GraphEncoder → LP head, and for a finetune the patient
    encoder and task head (README §3.1)."""

    def __init__(self, spec: ModelSpec, view: ViewShape):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.view = view
        self.input = InputModule(spec.input, view.num_nodes, len(view.node_types))
        self.encoder = _build_encoder(spec, view)
        self.lp_head = DistMultHead(spec.lp_head, len(view.relations), self.encoder.width)
        self.patient_encoder: Optional[PatientEncoder] = None
        self.task_head: Optional[TaskHead] = None
        if spec.learning_stage == "finetune":
            self.patient_encoder = PatientEncoder(spec.patient_encoder, self.encoder.width)
            self.task_head = build_task_head(spec.task_head, self.encoder.width)

    @property
    def width(self) -> int:
        return self.encoder.width

    @property
    def task(self) -> Optional[str]:
        return self.spec.task

    def node_vectors_sampled(self, node_ids: Tensor, edge_index: Tensor,
                             nodes_per_hop: Sequence[int], edges_per_hop: Sequence[int], *,
                             edge_type: Optional[Tensor] = None,
                             node_type: Optional[Tensor] = None) -> Tensor:
        """Vectors of a sampled subgraph's seeds (README §2.8)."""
        x = self.input(node_ids, node_type)
        return self.encoder.forward_sampled(x, edge_index, nodes_per_hop, edges_per_hop,
                                            edge_type=edge_type, node_type=node_type)

    @torch.no_grad()
    def node_vectors_full(self, edge_index: Tensor, *, edge_type: Optional[Tensor] = None,
                          node_type: Optional[Tensor] = None, chunk_size: int = 65536) -> Tensor:
        """Vectors of every view node by the chunked full-graph rule (§2.8).
        Call in evaluation mode."""
        if self.training:
            raise RuntimeError("full-view inference runs in evaluation mode")
        ids = torch.arange(self.view.num_nodes, device=edge_index.device)
        x = self.input(ids, node_type)
        return self.encoder.forward_full(x, edge_index, edge_type=edge_type, node_type=node_type,
                                         chunk_size=chunk_size)


def _build_encoder(spec: ModelSpec, view: ViewShape) -> GraphEncoder:
    kind = spec.encoder.kind
    if kind == "gatv2":
        return GATv2Encoder(spec.encoder, spec.input.dim, len(view.relations))
    raise SpecError(f"encoder kind {kind!r} is declared but not yet behind the GraphEncoder "
                    "contract in this build")


def build_bundle(spec: ModelSpec, view: Any) -> ModelBundle:
    """Build from a spec and a view (a `GraphView`, its record, or a `ViewShape`)."""
    if isinstance(view, ViewShape):
        shape = view
    elif isinstance(view, Mapping):
        shape = ViewShape.from_record(view)
    else:
        shape = ViewShape.from_record(view.record())
    return ModelBundle(spec, shape)


__all__ = ["ModelBundle", "ViewShape", "build_bundle"]
