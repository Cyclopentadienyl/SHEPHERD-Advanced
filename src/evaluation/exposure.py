"""
A KG link-prediction run's exposure: its measured edges against the model's recorded graph use.
================================================================================================
Research comparison §5 Stage 5 (the plan review's F4 and its §5 items 5 and 9).
A KG edge set's run (`task_measurement.run_kg_link_prediction`) keeps, beside
its metrics, how the edges it measured stand to what the measured model's own
record says it used of the graph: the report's top-level ``exposure`` block,
built here once from that record and the version, and kept with the result.

**Scoped to the recorded evidence.** The words say what the model's record
(its ``graph_exposure``, read in its writer's words by
`src.models.bundle.record.recorded_graph_use`, and its recorded edge split,
rebuilt from the view) and the KG edge set (its split, role and pairs) show,
and nothing more:

- a base measured on its own split, none of whose measured edges its record
  put in graph use: "held out relative to the recorded training and
  validation graph use" (a ``test`` set of a base);
- a model whose record passed messages over the full view (a finetune's
  validation, the gene task's distances): "after full-view exposure", whatever
  part the measured edges are in;
- a model of the same view whose recorded split differs from the version's:
  how many of the measured edges were in that model's recorded graph use,
  "k of n";
- on its own split with some measured edges in its recorded use (a ``train``
  or ``val`` set): "k of n" likewise;
- a record whose exposure or split this build cannot read: "exposure not
  recorded", never "unseen".

It never establishes that no other exposure happened (another run, a
person's choice, a file outside the record), and **it is never a gate**: no
measurement, registration, query or display is refused or hidden by it. The
negatives of a KG edge set are rule-generated, never proven non-relations
(`EXPOSURE_CAVEAT`).

An edge is a directed edge of the graph view, ``(head, relation, tail)`` in
the view's node and relation ids (`edge_keys`); a measured edge is the
positive of a pairs-file row.

Module: src/evaluation/exposure.py
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

import numpy as np

# The definition's version, its states, their readings, the caveat every
# block states and the counts' unit: the words a surface renders too, so they
# are kept once, in `src.evaluation.caveats` (standard library only), and
# written into each block from there.
from src.evaluation.caveats import (
    AFTER_FULL_VIEW_EXPOSURE,
    ANOTHER_SPLIT,
    EXPOSURE_CAVEAT,
    EXPOSURE_DEFINITION_VERSION,
    EXPOSURE_READINGS,
    EXPOSURE_STATES,
    HELD_OUT,
    IN_RECORDED_GRAPH_USE,
    MEASURED_EDGE_UNIT,
    NOT_RECORDED,
)

#: The model-record fields an exposure block reads.
RECORD_FIELDS_READ = ("graph_exposure", "data_roles.split")


def edge_keys(view: Any, head: Any, relation: Any, tail: Any) -> np.ndarray:
    """One int64 per directed edge ``(head, relation, tail)`` of `view` (view
    node and relation ids): equal exactly for the same edge, whichever part it
    is in."""
    nodes, relations = int(view.num_nodes), max(len(view.relations), 1)
    h, r, t = (np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=np.int64)
               for x in (head, relation, tail))
    return (h * relations + r) * nodes + t


def measured_in_parts(view: Any, split: Any, pairs: Any) -> Dict[str, np.ndarray]:
    """For each part of `split` (an `EdgeSplit` of `view`), whether each
    measured edge (each pairs row's positive) is an edge of that part."""
    from src.kg.graph_view import SPLIT_NAMES

    measured = edge_keys(view, pairs.head, pairs.relation, pairs.tail)
    found: Dict[str, np.ndarray] = {}
    for part in SPLIT_NAMES:
        mask = split.mask(part)
        found[part] = np.isin(measured, edge_keys(view, view.src[mask], view.rel[mask],
                                                  view.dst[mask]))
    return found


def kg_edges_exposure(*, record: Any, checkpoint_sha256: str, version: Mapping[str, Any],
                      view: Any, version_split: Any, pairs: Any,
                      in_version: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """The exposure block of a run that measured the KG edge set `version`
    (its report block, `cohort_registry.cohort_version_block`), whose split
    `version_split` was rebuilt from `view` and whose measured edges are in
    that split's parts as `in_version` says (`measured_in_parts`), with the
    model whose record `record` (`ModelRecord`) the checkpoint
    `checkpoint_sha256` holds.

    Never raises for what the record says: a record whose exposure or split
    this build cannot read gives ``not_recorded`` with the reason.
    """
    from src.kg.graph_view import SPLIT_NAMES, GraphViewError, rebuild_split
    from src.models.bundle.record import EXPOSURE_ALL_PARTS, recorded_graph_use

    n = int(pairs.head.numel())
    in_message_passing = int(in_version["train"].sum())
    model_split_record = record.data_roles.get("split")
    block: Dict[str, Any] = {
        "definition_version": EXPOSURE_DEFINITION_VERSION,
        "state": NOT_RECORDED,
        "reading": EXPOSURE_READINGS[NOT_RECORDED],
        "reason": None,
        "sources": {
            "model_record": {
                "checkpoint_sha256": checkpoint_sha256,
                "learning_stage": record.learning_stage,
                "operation": record.operation,
                "fields_read": list(RECORD_FIELDS_READ),
                "graph_exposure": None,
                "split_digest": (model_split_record.get("digest")
                                 if isinstance(model_split_record, Mapping) else None),
            },
            "version": {
                "label": version["label"], "kind": version["kind"], "role": version["role"],
                "manifest_sha256": version["manifest_sha256"],
                "split_digest": version["split"]["digest"],
            },
        },
        "split": None,
        "measured_edges": {"n": n, "unit": MEASURED_EDGE_UNIT, "in_model_parts": None,
                           "in_recorded_graph_use": None, "recorded_graph_use": None},
        # The run's own message passing (the version's, its split's train
        # part): its measured edges are in it exactly for role train.
        "measurement_message_passing": {
            "graph": "the KG edge set's message-passing graph: its split's train part",
            "measured_edges_in_it": in_message_passing,
            "includes_measured_edges": bool(version["message_passing"]["includes_measured_edges"]),
        },
        "caveat": EXPOSURE_CAVEAT,
    }

    def not_recorded(reason: str) -> Dict[str, Any]:
        block["reason"] = reason
        return block

    try:
        uses = recorded_graph_use(record.graph_exposure)
    except ValueError as exc:
        return not_recorded(str(exc))
    try:
        model_split = rebuild_split(view, model_split_record or {})
    except GraphViewError as exc:
        return not_recorded(f"the model record's edge split does not rebuild from this view: "
                            f"{exc}")
    block["sources"]["model_record"]["graph_exposure"] = copy.deepcopy(dict(record.graph_exposure))
    same = model_split.digest == version_split.digest
    in_model = in_version if same else measured_in_parts(view, model_split, pairs)
    used_parts = sorted({part for parts in uses.values() for part in parts},
                        key=SPLIT_NAMES.index)
    used = np.zeros(n, dtype=bool)
    for part in used_parts:
        used |= in_model[part]
    k = int(used.sum())
    full_view = sorted(at for at, parts in uses.items() if tuple(parts) == EXPOSURE_ALL_PARTS)
    block["split"] = "the version's split" if same else "another split of the same view"
    block["measured_edges"].update({
        "in_model_parts": {part: int(in_model[part].sum()) for part in SPLIT_NAMES},
        "in_recorded_graph_use": k,
        "recorded_graph_use": {at: list(parts) for at, parts in uses.items()},
    })
    if full_view:
        state = AFTER_FULL_VIEW_EXPOSURE
        block["reason"] = (f"the model record's {', '.join(full_view)} covered the full view, so "
                           "every measured edge was in that graph use")
    elif not same:
        state = ANOTHER_SPLIT
    else:
        state = HELD_OUT if k == 0 else IN_RECORDED_GRAPH_USE
    block["state"] = state
    block["reading"] = EXPOSURE_READINGS[state].format(k=k, n=n)
    return block


__all__ = ["AFTER_FULL_VIEW_EXPOSURE", "ANOTHER_SPLIT", "EXPOSURE_CAVEAT",
           "EXPOSURE_DEFINITION_VERSION", "EXPOSURE_READINGS", "EXPOSURE_STATES", "HELD_OUT",
           "IN_RECORDED_GRAPH_USE", "MEASURED_EDGE_UNIT", "NOT_RECORDED", "RECORD_FIELDS_READ",
           "edge_keys", "kg_edges_exposure", "measured_in_parts"]
