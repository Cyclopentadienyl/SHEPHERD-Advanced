"""
Evaluation protocols: which new measurement can run now, judged in one place.
==============================================================================
A protocol is a measurement this build implements. It is keyed by its own
name, which the command line takes as `--mode` and a report records as its
manifest's `mode`, and it states the model record it takes (a learning stage
and a task head) and the evaluator that runs it. The requested protocol and
the model's training task are separate: one model can be measured by several
protocols, and a protocol's name need not equal any task.

**Only new measurements are judged here.** The command line
(`scripts/measure_scorer.py`), the job runner (`TestRunInputs`, `start_test`)
and the Test page's Start ask this module whether a measurement can run now
and say its reason when it cannot. No reader, registration, query, display or
export uses it: a kept report is read under the mode it records, whether or
not that mode is a protocol here.

The judgment is computed from the model record the model spec writes (its
learning stage and task), a registered cohort version's counts and this
table; nothing else is stored. Every function reads `PROTOCOLS` when it is
called, so no consumer holds a copy taken at import.

Standard library only: the WebUI and the job runner import it without torch.

Module: src/evaluation/protocols.py
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Protocol:
    """One measurement this build implements, and the model it takes."""

    #: The command line's `--mode` and the report's `manifest.mode`.
    name: str
    #: The model record's learning stage it takes.
    learning_stage: str
    #: The task head it needs, `load_bundle`'s expected task (None: no head).
    task: str | None
    #: Its run and semantics: a key of `task_measurement.TASK_RUNS` and
    #: `task_measurement.task_semantics`.
    evaluator: str
    #: Cases are read with their gene fields, and a generated split is measured
    #: through the workspace's functional fixture.
    gene_fields: bool = False
    #: The `counts["genes"]` entries a registered cohort version must have
    #: above 0 (the importer writes them only for a source that lists genes).
    needs: tuple[str, ...] = ()


#: The measurements this build implements, by name, in the order they are offered.
PROTOCOLS: dict[str, Protocol] = {p.name: p for p in (
    Protocol("disease", "finetune", "disease", "disease"),
    Protocol("gene", "finetune", "gene", "gene", gene_fields=True,
             needs=("cases_with_causal_genes", "cases_with_candidate_genes")),
    Protocol("patients_like_me", "finetune", "patients_like_me", "patients_like_me",
             gene_fields=True, needs=("cases_with_causal_genes",)),
)}

#: The reason for a base model while no protocol takes a base. It never says
#: that a base has nothing to test: its formal test is not implemented yet.
BASE_MESSAGE = ("This model is a base: its formal test (link prediction on held-out "
                "knowledge-graph edges) is not implemented yet. A model finetuned from it can "
                "be measured.")

#: The reason for a listing whose checkpoint has no model record this build
#: reads (no learning stage).
NO_RECORD = "this checkpoint has no model record this build reads."

#: What a cohort version lacks when a need is missing, in the words a reason
#: uses: the genes it lists none of, and the source field that lists them.
_NEED_WORDS: dict[str, tuple[str, str]] = {
    "cases_with_causal_genes": ("causal", "causal_genes"),
    "cases_with_candidate_genes": ("candidate", "candidate_genes"),
}


def names() -> tuple[str, ...]:
    """The protocol names, in definition order."""
    return tuple(PROTOCOLS)


def _head(task: str | None) -> str:
    return "no task head" if task is None else f"the {task} head"


def _unknown(name: Any) -> str:
    return f"{name!r} is not a measurement this build implements ({', '.join(PROTOCOLS)})."


def _no_protocol_takes(learning_stage: str | None) -> str | None:
    """The reason for a record no protocol can take whatever its head: none
    read (no stage), or a base while no protocol takes a base; else None."""
    if learning_stage is None:
        return NO_RECORD
    if learning_stage == "base" and not any(p.learning_stage == "base"
                                            for p in PROTOCOLS.values()):
        return BASE_MESSAGE
    return None


def available(learning_stage: str | None,
              task: str | None) -> tuple[tuple[str, ...], str | None]:
    """The protocols whose model requirement a record of `learning_stage` and
    `task` meets, in definition order, and None; or no protocol and the reason
    none does (None for both when no record was read)."""
    found = tuple(p.name for p in PROTOCOLS.values()
                  if p.learning_stage == learning_stage and p.task == task)
    if found:
        return found, None
    return (), (_no_protocol_takes(learning_stage)
                or f"no measurement this build implements takes a {learning_stage} model "
                   f"with {_head(task)}.")


def model_refusal(name: Any, learning_stage: str | None,
                  task: str | None) -> str | None:
    """Why the protocol `name` cannot measure a model whose record has
    `learning_stage` and `task` (None for both when no record was read), or
    None. In this order: a name that is not a protocol; no readable record; a
    base while no protocol takes a base; a stage or head that differs, with the
    protocols the model can be measured by, if any."""
    protocol = PROTOCOLS.get(name) if isinstance(name, str) else None
    if protocol is None:
        return _unknown(name)
    if protocol.learning_stage == learning_stage and protocol.task == task:
        return None
    reason = _no_protocol_takes(learning_stage)
    if reason is not None:
        return reason
    said = (f"the {protocol.name} measurement takes a {protocol.learning_stage} model with "
            f"{_head(protocol.task)}; this checkpoint is a {learning_stage} model with "
            f"{_head(task)}.")
    found = available(learning_stage, task)[0]
    return f"{said} It can be measured by {', '.join(found)}." if found else said


def inputs_refusal(name: Any, counts: Mapping[str, Any] | None) -> str | None:
    """Why a registered cohort version whose `counts` (its `cohort.json`'s)
    are these cannot be measured by the protocol `name`, or None: the first of
    its needs the version's `counts["genes"]` does not have above 0. A name
    that is not a protocol gets the same reason as from `model_refusal`."""
    protocol = PROTOCOLS.get(name) if isinstance(name, str) else None
    if protocol is None:
        return _unknown(name)
    genes = counts.get("genes") if isinstance(counts, Mapping) else None
    genes = genes if isinstance(genes, Mapping) else {}
    for need in protocol.needs:
        value = genes.get(need)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            continue
        if need in _NEED_WORDS:
            kind, field = _NEED_WORDS[need]
            return (f"this cohort version lists no {kind} genes, which the {protocol.name} "
                    f"measurement needs. Import a source that lists {field}.")
        return (f"this cohort version records no {need}, which the {protocol.name} "
                "measurement needs.")
    return None


def refusal(name: Any, learning_stage: str | None, task: str | None,
            counts: Mapping[str, Any] | None) -> str | None:
    """`model_refusal`, then `inputs_refusal`: the one reason for Start's
    preview and for the start itself, or None."""
    return model_refusal(name, learning_stage, task) or inputs_refusal(name, counts)


__all__ = ["BASE_MESSAGE", "NO_RECORD", "PROTOCOLS", "Protocol", "available", "inputs_refusal",
           "model_refusal", "names", "refusal"]
