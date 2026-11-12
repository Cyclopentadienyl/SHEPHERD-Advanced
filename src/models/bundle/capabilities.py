"""
A model's capabilities: what it offers a measurement, and what it needs.
=======================================================================
`docs/working/research-comparison/README.md` §5 Stage 3, §6 E2 and E5.

**One description per model, derived from its validated spec** and from the
components the one builder (`src.models.bundle.builder`) builds from it:
the input module, the encoder as its kind declares it
(`src.models.bundle.spec.EncoderDeclaration`), the LP head, and for a
finetune the patient encoder and the task head. `model_capabilities` says:

- `learning_stage`: the spec's, for the words a reason uses only;
- `encoder`: the encoder's kind;
- `representations`: what it outputs: `node_vectors` (the encoder's), and a
  `patient_representation` where a patient encoder pools them (a finetune);
- `heads`: each head it carries, as `(kind, name, scoring semantics)`: the LP
  head (`LINK_PREDICTION`, its spec kind, its semantics stated by
  `LP_HEAD_SEMANTICS`, never assumed from the attribute name `lp_head`) and
  each task head (`TASK`, its task, `TASK_HEAD_SEMANTICS`). A head whose
  semantics this build does not state is listed with None: no protocol scores
  with it, and the description is still given, so a list never fails on it.
  A model offers a set of heads; nothing here assumes it carries one task;
- `node_inputs`: `node_ids` (its learned table's rows) and `node_types` where
  its input adds a type embedding or its encoder reads them;
- `relation_inputs`: `relation_ids` (the LP head scores each pair under its
  relation's vector) and `edge_relation_ids` where its encoder reads each
  message-passing edge's relation;
- `computation`: the computations its encoder declares (`sampled`,
  `full_view`), and `sampled_hops`, the hops its sampled computation needs;
- `node_vector_width`: one node vector's width, as the encoder declares it.

An encoder kind this build does not implement declares nothing: its model
offers no computation, its hops and width are None, and a measurement that
needs a computation refuses it before anything is built, as the start of a
training run of it does (`hops_problem`); the builder refuses it too.

**Plain data**: strings, integers, None and tuples (sorted where the order
means nothing), so `src.evaluation.protocols` judges it with the standard
library alone, every list carries it as the one listing reader gives it
(`src.models.bundle.record.listed_record`) and its JSON is the same on every
read. Torch-free, as the spec is.

Module: src/models/bundle/capabilities.py
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.models.bundle.spec import (
    FULL_VIEW,
    SAMPLED,
    ModelSpec,
    encoder_declaration,
)

#: The representations a model outputs.
NODE_VECTORS = "node_vectors"
PATIENT_REPRESENTATION = "patient_representation"

#: The kinds of head a description lists.
LINK_PREDICTION = "link_prediction"
TASK = "task"

#: The LP head's scoring semantics, by its spec kind (`LPHeadSpec.kind`): the
#: DistMult raw score `Σ h ⊙ w_r ⊙ t` (decision raw > 0), trained through tanh
#: by a margin hinge (`src.models.bundle.lp_head`, README A2).
LP_HEAD_SEMANTICS: dict[str, str] = {"distmult_tanh": "distmult_raw_score/tanh_hinge/v1"}

#: Each task head's scoring semantics, by its task (`src.models.tasks.heads`):
#: disease and patients-like-me score the negative normalised squared distance
#: between the projected patient and each projected candidate (a disease, or
#: another patient through the same patient tower); gene scores a case's own
#: gene list by EMBSIM mixed with SPLSIM (README §6).
TASK_HEAD_SEMANTICS: dict[str, str] = {
    "disease": "patient_disease/negative_normalised_squared_distance/v1",
    "gene": "patient_gene_list/embsim_with_splsim/v1",
    "patients_like_me": "patient_patient/negative_normalised_squared_distance/v1",
}

#: The inputs a model can need besides its message-passing edges.
NODE_IDS = "node_ids"
NODE_TYPES = "node_types"
RELATION_IDS = "relation_ids"
EDGE_RELATION_IDS = "edge_relation_ids"

#: The computations an encoder can declare (`src.models.bundle.spec`).
COMPUTATIONS = (FULL_VIEW, SAMPLED)

#: The keys of a description, in the order `model_capabilities` gives them.
CAPABILITY_FIELDS = ("learning_stage", "encoder", "representations", "heads", "node_inputs",
                     "relation_inputs", "computation", "sampled_hops", "node_vector_width")


def _semantics(table: Mapping[str, str], name: str) -> str | None:
    """The scoring semantics `table` states for `name`, or None: a head whose
    semantics this build does not state is still described (so a list that
    carries the description never fails on it), and no protocol scores with it."""
    return table.get(name)


def model_capabilities(spec: ModelSpec) -> dict[str, Any]:
    """The capability description of a model of `spec` (validated first): see
    the module's docstring for each field. The same spec gives the same
    description, field for field."""
    spec.validate()
    declaration = encoder_declaration(spec.encoder.kind)
    heads: tuple[tuple[str, str, str | None], ...] = ((
        LINK_PREDICTION, spec.lp_head.kind,
        _semantics(LP_HEAD_SEMANTICS, spec.lp_head.kind)),)
    representations = {NODE_VECTORS}
    if spec.patient_encoder is not None:
        representations.add(PATIENT_REPRESENTATION)
    if spec.task_head is not None:
        task = spec.task_head.task
        heads += ((TASK, task, _semantics(TASK_HEAD_SEMANTICS, task)),)
    node_inputs = {NODE_IDS}
    relation_inputs = {RELATION_IDS}
    if spec.input.type_embedding:
        node_inputs.add(NODE_TYPES)
    if declaration is not None:
        if declaration.reads_node_types(spec.encoder):
            node_inputs.add(NODE_TYPES)
        if declaration.reads_edge_relations(spec.encoder):
            relation_inputs.add(EDGE_RELATION_IDS)
    computation = tuple(sorted(declaration.computation)) if declaration is not None else ()
    return {
        "learning_stage": spec.learning_stage,
        "encoder": spec.encoder.kind,
        "representations": tuple(sorted(representations)),
        "heads": heads,
        "node_inputs": tuple(sorted(node_inputs)),
        "relation_inputs": tuple(sorted(relation_inputs)),
        "computation": computation,
        "sampled_hops": (declaration.sampled_hops(spec.encoder)
                         if declaration is not None and SAMPLED in computation else None),
        "node_vector_width": declaration.width(spec.encoder) if declaration is not None else None,
    }


__all__ = ["CAPABILITY_FIELDS", "COMPUTATIONS", "EDGE_RELATION_IDS", "LINK_PREDICTION",
           "LP_HEAD_SEMANTICS", "NODE_IDS", "NODE_TYPES", "NODE_VECTORS",
           "PATIENT_REPRESENTATION", "RELATION_IDS", "TASK", "TASK_HEAD_SEMANTICS",
           "model_capabilities"]
