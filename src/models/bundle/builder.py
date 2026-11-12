"""
The one builder: a model spec and a graph view's shape → one `nn.Module`.
========================================================================
`docs/working/scorer-retraining/README.md` §3.1, §4.

Every trainable parameter (node table, encoder, relation vectors, patient
encoder, task head) is a submodule of the returned `ModelBundle`. The builder
reads the spec and the view's shape only: no weight name, state-dict shape or
training flag decides what is built. The Trainer's optimizer, the checkpoint
writer and reader, serving, measurement and the scripts all build through here.

**Its parts** (`PARTS`: the graph side, then a finetune's downstream parts)
are the names its weights carry first. **A frozen part** (`ModelBundle.freeze`,
research comparison Stage 9, owner's O4a) has no parameter that trains, so the
optimizer, built after the freeze, holds none of it, and it runs in evaluation
mode for the bundle's whole life: `train()` sets every other part's mode and
leaves a frozen one in eval, so no caller of `train()` switches it back.
BatchNorm then uses its running statistics and never updates them, and
dropout is off, whatever the run does around it.

Module: src/models/bundle/builder.py
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bundle.encoders import GATv2Encoder, GraphEncoder, InputModule
from src.models.bundle.lp_head import DistMultHead
from src.models.bundle.spec import (ModelSpec, SpecError, ViewShape, encoder_declaration,
                                    encoder_not_built)
from src.models.tasks.heads import PatientEncoder, TaskHead, build_task_head


#: The graph side, every bundle's parts: the input module, the encoder and the
#: LP head, by their attribute names (the first part of their weights' names).
GRAPH_SIDE_PARTS = ("input", "encoder", "lp_head")
#: A finetune's downstream parts, after its graph side: the patient encoder and
#: the task head.
DOWNSTREAM_PARTS = ("patient_encoder", "task_head")
#: Every part a bundle can have, in its order.
PARTS = GRAPH_SIDE_PARTS + DOWNSTREAM_PARTS
#: The forward mode of a frozen part, for the bundle's whole life.
FROZEN_FORWARD = "eval"
#: How `ModelBundle.parts_sha256` hashes parts' state, named where a record
#: keeps such a digest: every state entry (parameter and buffer) whose name's
#: first part is one of the parts, by name in sorted order; for each, its name,
#: dtype and shape as JSON (keys sorted), then its bytes as stored (on the CPU,
#: contiguous, in the platform's byte order).
PARTS_DIGEST_RULE = "parts_state_sha256/1"


class ModelBundle(nn.Module):
    """Input module → GraphEncoder → LP head, and for a finetune the patient
    encoder and task head (README §3.1)."""

    def __init__(self, spec: ModelSpec, view: ViewShape):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.view = view
        # Before any submodule: `train()` reads it.
        self._frozen: Tuple[str, ...] = ()
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
    def parts(self) -> Tuple[str, ...]:
        """The parts this bundle has, in `PARTS`' order: the graph side, and a
        finetune's downstream parts."""
        return tuple(part for part in PARTS if getattr(self, part) is not None)

    @property
    def frozen(self) -> Tuple[str, ...]:
        """The parts `freeze` keeps from training, in the bundle's order."""
        return self._frozen

    def freeze(self, parts: Collection[str]) -> None:
        """**Keep `parts` from training for the bundle's whole life** (research
        comparison Stage 9, owner's O4a): every parameter of each has
        `requires_grad` False, so an optimizer built after this holds none of
        them, and each runs in evaluation mode now and after every `train()`
        (`FROZEN_FORWARD`). Called once, before the optimizer is built; an empty
        `parts` freezes nothing. A part this bundle does not have, or a freeze
        that leaves no part to train, is refused (a `ValueError`); nothing is
        changed then."""
        if self._frozen:
            raise ValueError(f"the bundle's parts are frozen already ({', '.join(self._frozen)})")
        unknown = sorted(set(parts) - set(self.parts))
        if unknown:
            raise ValueError(f"this bundle has no part {', '.join(unknown)} to freeze (its parts: "
                             f"{', '.join(self.parts)})")
        frozen = tuple(part for part in self.parts if part in set(parts))
        if frozen and frozen == self.parts:
            raise ValueError(f"freezing {', '.join(frozen)} leaves no part of this bundle to train")
        for part in frozen:
            getattr(self, part).requires_grad_(False)
        self._frozen = frozen
        self.train(self.training)

    def train(self, mode: bool = True) -> ModelBundle:
        """`nn.Module.train`, every part set to `mode` but a frozen one, which
        stays in evaluation mode (`freeze`): whoever calls `train()` (the
        Trainer each epoch, `eval()`, any other caller), a frozen part's
        BatchNorm never updates its running statistics and its dropout stays
        off."""
        super().train(mode)
        for part in self._frozen:
            getattr(self, part).eval()
        return self

    def parts_sha256(self, parts: Collection[str]) -> str:
        """The SHA-256 of `parts`' state as it is now, by `PARTS_DIGEST_RULE`:
        two bundles of one spec whose parts hold the same values give the same
        digest, and any changed value, buffer included, changes it. A part
        this bundle does not have is refused (a `ValueError`)."""
        unknown = sorted(set(parts) - set(self.parts))
        if unknown:
            raise ValueError(f"this bundle has no part {', '.join(unknown)} (its parts: "
                             f"{', '.join(self.parts)})")
        state = self.state_dict()
        digest = hashlib.sha256()
        for name in sorted(key for key in state if key.split(".", 1)[0] in set(parts)):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(json.dumps({"name": name, "dtype": str(tensor.dtype),
                                      "shape": list(tensor.shape)}, sort_keys=True).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True,
                        assign: bool = False):
        """`nn.Module.load_state_dict`, after **one check every load of a
        bundle's weights passes** (serving and measurement through
        `load_bundle`, a finetune's or a branch's parent weights, a resume's
        restore): weights the spec makes one set are one set in the file.

        A spec that shares parameters (`encoder.share_weights`: a GATv2
        layer's source and target transforms are one module) names each
        shared tensor under several keys. torch copies every key into the
        one tensor, the last copy winning, so a file holding two different
        values under those keys used to load as whichever came last (the
        reviewer's K8-1 recheck). Such a file is not this model: refused, as
        torch refuses a state that does not fit, with a `RuntimeError` naming
        the keys, before anything is copied."""
        _refuse_contradictory_shared_weights(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def task(self) -> Optional[str]:
        return self.spec.task

    def node_vectors_sampled(self, node_ids: Tensor, edge_index: Tensor,
                             nodes_per_hop: Sequence[int], edges_per_hop: Sequence[int], *,
                             edge_type: Optional[Tensor] = None,
                             node_type: Optional[Tensor] = None) -> Tensor:
        """Vectors of a sampled subgraph's seeds (README §2.8).

        `node_ids` are view ids in the sampler's hop order; `node_type` holds
        each one's view type id in that order, and `edge_type` each
        message-passing edge's relation id in `edge_index`'s column order. The
        training steps call this through `LPBatch.node_vectors`, whose sampler
        reads both from the graph view."""
        x = self.input(node_ids, node_type)
        return self.encoder.forward_sampled(x, edge_index, nodes_per_hop, edges_per_hop,
                                            edge_type=edge_type, node_type=node_type)

    @torch.no_grad()
    def node_vectors_full(self, edge_index: Tensor, *, edge_type: Optional[Tensor] = None,
                          node_type: Optional[Tensor] = None, chunk_size: int = 65536) -> Tensor:
        """Vectors of every view node by the chunked full-graph rule (§2.8).
        Call in evaluation mode.

        The nodes are the view's, in its node order; `node_type` holds each
        one's view type id in that order (the view's `GraphView.node_type`),
        and `edge_type` each edge's relation id in `edge_index`'s column order.
        Validation, measurement and serving call this through
        `src.models.tasks.scoring.full_view_vectors`."""
        if self.training:
            raise RuntimeError("full-view inference runs in evaluation mode")
        ids = torch.arange(self.view.num_nodes, device=edge_index.device)
        x = self.input(ids, node_type)
        return self.encoder.forward_full(x, edge_index, edge_type=edge_type, node_type=node_type,
                                         chunk_size=chunk_size)


def _refuse_contradictory_shared_weights(module: nn.Module, state: Mapping[str, Any]) -> None:
    """Refuse `state` when two of its keys that name one tensor of `module`
    (a shared parameter or buffer) hold different values. A key the state
    lacks, or a value that is not a tensor, is left to the load itself."""
    keys_by_tensor: Dict[int, List[str]] = {}
    for key, tensor in module.state_dict(keep_vars=True).items():
        keys_by_tensor.setdefault(id(tensor), []).append(key)
    for keys in keys_by_tensor.values():
        given = [key for key in keys if isinstance(state.get(key), Tensor)]
        for other in given[1:]:
            first, second = state[given[0]], state[other]
            same = (first.shape == second.shape and first.dtype == second.dtype
                    and (torch.allclose(first, second, rtol=0.0, atol=0.0, equal_nan=True)
                         if first.is_floating_point() else torch.equal(first, second)))
            if not same:
                raise RuntimeError(
                    f"the weights hold two different values for one shared parameter "
                    f"({given[0]} and {other}): the model spec makes them one set, so this "
                    "state is not a model it describes")


#: The encoder implementation of each kind this build builds. A kind is built
#: only when it is declared (`src.models.bundle.spec.ENCODER_DECLARATIONS`,
#: what its capabilities and checks are read from) and implemented here.
_ENCODERS: Dict[str, type] = {"gatv2": GATv2Encoder}


def _build_encoder(spec: ModelSpec, view: ViewShape) -> GraphEncoder:
    """The spec's encoder, dispatched by its kind; any kind without both a
    declaration and an implementation is refused (`encoder_not_built`)."""
    kind = spec.encoder.kind
    implementation = _ENCODERS.get(kind) if encoder_declaration(kind) is not None else None
    if implementation is None:
        raise SpecError(encoder_not_built(kind))
    return implementation(spec.encoder, spec.input.dim, len(view.relations))


def build_bundle(spec: ModelSpec, view: Any) -> ModelBundle:
    """Build from a spec and a view (a `GraphView`, its record, or a `ViewShape`)."""
    if isinstance(view, ViewShape):
        shape = view
    elif isinstance(view, Mapping):
        shape = ViewShape.from_record(view)
    else:
        shape = ViewShape.from_record(view.record())
    return ModelBundle(spec, shape)


__all__ = ["DOWNSTREAM_PARTS", "FROZEN_FORWARD", "GRAPH_SIDE_PARTS", "ModelBundle", "PARTS",
           "PARTS_DIGEST_RULE", "ViewShape", "build_bundle"]
