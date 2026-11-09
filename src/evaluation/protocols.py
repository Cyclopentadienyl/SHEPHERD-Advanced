"""
Evaluation protocols: which new measurement can run now, judged in one place.
==============================================================================
A protocol is a measurement this build implements. It is keyed by its own
name, which the command line takes as `--mode` and a report records as its
manifest's `mode`, and it states what it needs of a model, the data kinds it
measures and the evaluator that runs it. The requested protocol and the
model's training task are separate: one model can be measured by several
protocols, and a protocol's name need not equal any task.

**A model is judged by its capabilities** (research comparison §5 Stage 3,
§6 E2 and E5). Every consumer judges the one capability description the
listing reader carries (`src.models.bundle.capabilities.model_capabilities`,
read through `src.models.bundle.record.listed_record` or
`ModelRecord.capabilities`): the representations a model outputs, the heads
it carries with their scoring semantics, the inputs it needs and the
computations its encoder declares. A protocol takes a model that carries the
head it scores with, outputs the representations and offers the computation
it reads, and whose inputs its evaluator supplies. No stage list, task name or
encoder kind decides; the learning stage and task head appear only in the
words a reason uses. A model offers a set of heads, so one model can be taken
by several protocols.

**Only new measurements are judged here.** The command line
(`scripts/measure_scorer.py`), the job runner (`TestRunInputs`, `start_test`),
the Test page's Start and the deployment probe ask this module whether a
measurement can run now and say its reason when it cannot. No reader,
registration, query, display or export uses it: a kept report is read under
the mode it records, whether or not that mode is a protocol here.

The judgment is computed from a model's capability description, the data
kind measured (`generated` for a generated split, a registered version's
recorded kind otherwise; the words of `src.evaluation.cohort.COHORT_KINDS`), a
registered version's counts and this table; nothing else is stored. Every
function reads `PROTOCOLS` when it is called, so no consumer holds a copy taken
at import.

Standard library only: the WebUI and the job runner import it without torch.

Module: src/evaluation/protocols.py
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The capability words this module reads in a description, as
#: `src.models.bundle.capabilities` writes them (pinned by tests/unit/test_protocols.py).
LINK_PREDICTION = "link_prediction"
TASK = "task"
NODE_VECTORS = "node_vectors"
PATIENT_REPRESENTATION = "patient_representation"
FULL_VIEW = "full_view"
#: The model inputs the task evaluators supply: each node's id and type, and
#: each message-passing edge's relation id (`full_view_vectors` over the graph
#: view), which also names the relations the LP head scores under.
FULL_VIEW_INPUTS = ("edge_relation_ids", "node_ids", "node_types", "relation_ids")

#: The data kinds measured (`src.evaluation.cohort.COHORT_KINDS`): a generated
#: split, and a registered version's recorded kind (a supplied cohort, a KG
#: edge set, a generated split kept in the registry).
GENERATED = "generated"
SUPPLIED = "supplied"
KG_EDGES = "kg_edges"

#: The reason a registered generated split is not measured yet: the registry
#: keeps and lists one (research comparison §5 Stage 4), and measuring it is
#: the round's Stage 5. A stated capability gap, said before anything is read,
#: never a judgment of its role; the workspace's own split is measured as
#: before.
GENERATED_VERSION_GAP = ("this build lists and reads a registered generated split but does not "
                         "measure one yet; the workspace's own generated split is measured by "
                         "the command line's --split.")

#: The task heads' scoring semantics, by task, as
#: `src.models.bundle.capabilities.TASK_HEAD_SEMANTICS` states them.
_TASK_SEMANTICS = {
    "disease": "patient_disease/negative_normalised_squared_distance/v1",
    "gene": "patient_gene_list/embsim_with_splsim/v1",
    "patients_like_me": "patient_patient/negative_normalised_squared_distance/v1",
}


def task_head(task: str) -> tuple[str, str, str]:
    """The task head of `task`, as a capability description lists it:
    `(TASK, task, its scoring semantics)`."""
    return (TASK, task, _TASK_SEMANTICS[task])


@dataclass(frozen=True)
class Protocol:
    """One measurement this build implements: what it needs of a model and
    of the data, and the evaluator that runs it."""

    #: The command line's `--mode` and the report's `manifest.mode`.
    name: str
    #: Its run and semantics: a key of `task_measurement.TASK_RUNS` and
    #: `task_measurement.task_semantics`.
    evaluator: str
    #: The head its evaluator scores with, `(kind, name, scoring semantics)`,
    #: as a model's capability description lists its heads.
    head: tuple[str, str, str]
    #: The representations its evaluator reads.
    representations: tuple[str, ...] = (NODE_VECTORS,)
    #: The computation its evaluator asks of the encoder.
    computation: str = FULL_VIEW
    #: The model inputs its evaluator supplies: a model that needs another is
    #: refused.
    supplies: tuple[str, ...] = FULL_VIEW_INPUTS
    #: The data kinds it measures.
    data_kinds: tuple[str, ...] = (GENERATED, SUPPLIED)
    #: Cases are read with their gene fields, and a generated split is measured
    #: through the workspace's functional fixture.
    gene_fields: bool = False
    #: The `counts["genes"]` entries a registered cohort version must have
    #: above 0 (the importer writes them only for a source that lists genes).
    #: A generated split has no counts; the run checks its fixture.
    needs: tuple[str, ...] = ()


#: What the three task protocols read of a model: node vectors, pooled by the
#: patient encoder into a patient representation, by the full-view rule.
_TASK_REPRESENTATIONS = (NODE_VECTORS, PATIENT_REPRESENTATION)

#: The measurements this build implements, by name, in the order they are offered.
PROTOCOLS: dict[str, Protocol] = {p.name: p for p in (
    Protocol("disease", "disease", task_head("disease"),
             representations=_TASK_REPRESENTATIONS),
    Protocol("gene", "gene", task_head("gene"), representations=_TASK_REPRESENTATIONS,
             gene_fields=True,
             needs=("cases_with_causal_genes", "cases_with_candidate_genes")),
    Protocol("patients_like_me", "patients_like_me", task_head("patients_like_me"),
             representations=_TASK_REPRESENTATIONS, gene_fields=True,
             needs=("cases_with_causal_genes",)),
)}

#: The reason for a base no protocol takes. It never says that a base has
#: nothing to test: its formal test is not implemented yet.
BASE_MESSAGE = ("This model is a base: its formal test (link prediction on held-out "
                "knowledge-graph edges) is not implemented yet. A model finetuned from it can "
                "be measured.")

#: The reason for a listing whose checkpoint has no model record this build
#: reads (so no capability description).
NO_RECORD = "this checkpoint has no model record this build reads."

#: What a cohort version lacks when a need is missing, in the words a reason
#: uses: the genes it lists none of, and the source field that lists them.
_NEED_WORDS: dict[str, tuple[str, str]] = {
    "cases_with_causal_genes": ("causal", "causal_genes"),
    "cases_with_candidate_genes": ("candidate", "candidate_genes"),
}

#: A capability or input as a reason names it.
_CAPABILITY_WORDS: dict[str, str] = {
    NODE_VECTORS: "node vectors",
    PATIENT_REPRESENTATION: "a patient representation",
    FULL_VIEW: "full-view computation",
    "sampled": "sampled computation",
    "node_ids": "node ids",
    "node_types": "node types",
    "relation_ids": "relation ids",
    "edge_relation_ids": "each message-passing edge's relation id",
}

#: A data kind as a reason names it.
_KIND_WORDS: dict[str, str] = {GENERATED: "a generated split",
                               SUPPLIED: "a supplied cohort version",
                               KG_EDGES: "a KG edge set"}


def names() -> tuple[str, ...]:
    """The protocol names, in definition order."""
    return tuple(PROTOCOLS)


def _unknown(name: Any) -> str:
    return f"{name!r} is not a measurement this build implements ({', '.join(PROTOCOLS)})."


def _protocol(name: Any) -> Protocol | None:
    return PROTOCOLS.get(name) if isinstance(name, str) else None


def _offered(model: Mapping[str, Any], key: str) -> frozenset:
    """The names a description lists under `key` (a list or tuple of names;
    anything else lists none)."""
    value = model.get(key)
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(v for v in value if isinstance(v, str))


def _heads(model: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """The heads a description lists, each as a tuple, in its order."""
    value = model.get("heads")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(tuple(h) for h in value
                 if isinstance(h, (list, tuple)) and all(isinstance(p, str) for p in h))


def _inputs(model: Mapping[str, Any]) -> frozenset:
    """Every input a description says its model needs."""
    return _offered(model, "node_inputs") | _offered(model, "relation_inputs")


def _accepts(protocol: Protocol, model: Any) -> bool:
    """Whether `protocol` takes the model `model` describes: it carries the
    head the protocol scores with, outputs the representations and offers the
    computation it reads, and needs no input its evaluator does not supply."""
    if not isinstance(model, Mapping):
        return False
    return (tuple(protocol.head) in _heads(model)
            and set(protocol.representations) <= _offered(model, "representations")
            and protocol.computation in _offered(model, "computation")
            and _inputs(model) <= set(protocol.supplies))


def _found(model: Any) -> tuple[str, ...]:
    """The protocols that take `model`, in definition order."""
    return tuple(p.name for p in PROTOCOLS.values() if _accepts(p, model))


def _takes(protocol: Protocol) -> str:
    """The model `protocol` takes, as a reason names it: by the head it scores
    with. A task head is a finetuned model's (the model spec gives a base
    none)."""
    kind, name, _semantics = protocol.head
    if kind == TASK:
        return f"a finetune model with the {name} head"
    return f"a model with the {name} head"


def _described(model: Mapping[str, Any]) -> str:
    """A model as a reason names it: its learning stage and its task heads."""
    tasks = [name for kind, name, *_rest in _heads(model) if kind == TASK]
    head = ("no task head" if not tasks
            else f"the {' and '.join(tasks)} head{'s' if len(tasks) > 1 else ''}")
    return f"a {model.get('learning_stage')} model with {head}"


def _words(names: Any) -> str:
    return ", ".join(_CAPABILITY_WORDS.get(n, n) for n in sorted(names))


def _unmet(protocol: Protocol, model: Mapping[str, Any]) -> str:
    """Why `protocol` does not take `model`, which carries the head it scores
    with: the representations or computation it lacks, then the inputs it
    needs that the protocol's evaluator does not supply."""
    lacks = set(protocol.representations) - _offered(model, "representations")
    if protocol.computation not in _offered(model, "computation"):
        lacks.add(protocol.computation)
    if lacks:
        return (f"the {protocol.name} measurement needs {_words(lacks)}, which this checkpoint's "
                f"model ({model.get('encoder')} encoder) does not offer in this build.")
    return (f"this checkpoint's model needs {_words(_inputs(model) - set(protocol.supplies))}, "
            f"which the {protocol.name} measurement does not supply.")


def _none_takes(model: Any, found: tuple[str, ...]) -> str | None:
    """The reason for a model no protocol can take, whatever is asked of it:
    no description (no record read), or a base no protocol takes; else None."""
    if not isinstance(model, Mapping):
        return NO_RECORD
    if not found and model.get("learning_stage") == "base":
        return BASE_MESSAGE
    return None


def available(model: Any) -> tuple[tuple[str, ...], str | None]:
    """The protocols that take the model whose capability description is
    `model` (None: no record read), in definition order, and None; or no
    protocol and the reason none does."""
    found = _found(model)
    if found:
        return found, None
    reason = _none_takes(model, found)
    if reason is not None:
        return (), reason
    for protocol in PROTOCOLS.values():
        if tuple(protocol.head) in _heads(model):
            return (), _unmet(protocol, model)
    return (), f"no measurement this build implements takes {_described(model)}."


def model_refusal(name: Any, model: Any) -> str | None:
    """Why the protocol `name` cannot measure the model whose capability
    description is `model` (None: no record read), or None. In this order: a
    name that is not a protocol; no description; a base no protocol takes; a
    model without the head the protocol scores with, then one lacking what
    else it needs, each with the protocols the model can be measured by, if
    any."""
    protocol = _protocol(name)
    if protocol is None:
        return _unknown(name)
    if _accepts(protocol, model):
        return None
    found = _found(model)
    reason = _none_takes(model, found)
    if reason is not None:
        return reason
    if tuple(protocol.head) not in _heads(model):
        said = (f"the {protocol.name} measurement takes {_takes(protocol)}; this checkpoint is "
                f"{_described(model)}.")
    else:
        said = _unmet(protocol, model)
    return f"{said} It can be measured by {', '.join(found)}." if found else said


def inputs_refusal(name: Any, kind: Any, counts: Mapping[str, Any] | None) -> str | None:
    """Why data of the kind `kind` (`GENERATED` for a generated split, a
    registered version's recorded kind otherwise) whose counts are `counts`
    (a registered version's `cohort.json` counts; None for a workspace's own
    generated split, which has none) cannot be measured by the protocol
    `name`, or None: a kind the protocol does not measure; a registered
    generated split, which this build does not measure yet
    (`GENERATED_VERSION_GAP`); then, for a registered version, the first of
    its needs the version's `counts["genes"]` does not have above 0. A name
    that is not a protocol gets the same reason as from `model_refusal`."""
    protocol = _protocol(name)
    if protocol is None:
        return _unknown(name)
    if kind not in protocol.data_kinds:
        said = _KIND_WORDS.get(kind, f"data of kind {kind!r}") if isinstance(kind, str) \
            else "data of no kind this build reads"
        takes = " or ".join(_KIND_WORDS.get(k, k) for k in protocol.data_kinds)
        return f"the {protocol.name} measurement does not measure {said}; it measures {takes}."
    if kind == GENERATED:
        return None if counts is None else GENERATED_VERSION_GAP
    genes = counts.get("genes") if isinstance(counts, Mapping) else None
    genes = genes if isinstance(genes, Mapping) else {}
    for need in protocol.needs:
        value = genes.get(need)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            continue
        if need in _NEED_WORDS:
            what, field = _NEED_WORDS[need]
            return (f"this cohort version lists no {what} genes, which the {protocol.name} "
                    f"measurement needs. Import a source that lists {field}.")
        return (f"this cohort version records no {need}, which the {protocol.name} "
                "measurement needs.")
    return None


def refusal(name: Any, model: Any, kind: Any, counts: Mapping[str, Any] | None) -> str | None:
    """`model_refusal`, then `inputs_refusal`: the one reason for Start's
    preview, for the start itself, for the command line and for the probe, or
    None."""
    return model_refusal(name, model) or inputs_refusal(name, kind, counts)


__all__ = ["BASE_MESSAGE", "FULL_VIEW_INPUTS", "GENERATED", "GENERATED_VERSION_GAP", "KG_EDGES",
           "NO_RECORD", "PROTOCOLS", "Protocol", "SUPPLIED", "available", "inputs_refusal",
           "model_refusal", "names", "refusal", "task_head"]
