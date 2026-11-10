"""
Capability descriptions for tests, by a record's learning stage and task.
========================================================================
Research comparison Stage 3: every consumer judges a new measurement by a
model's capability description (`src.models.bundle.capabilities`), as the
one listing reader carries it (`listed_record`). A test that states a model
by its learning stage and task head gets that model's description here:

- None for no record read (no description);
- for a record the model spec can write (a base with no task; a finetune
  with one of `spec.TASKS`), its reference spec's description, by the one
  derivation (`model_capabilities`);
- for one it cannot write (a finetune without a task head or with an unknown
  one, a base carrying a task head), a test double: the reference
  description of its stage (a finetune's of the disease head), with its task
  heads replaced by the head `task` names (none for None; an unknown task's
  semantics is its name), and nothing else changed;
- for `LP_HEAD_UNSTATED` (a base whose LP head declares scoring semantics
  this build does not state, so no protocol takes it: since the common KG
  link-prediction protocol, Stage 5, takes every model whose LP head this
  build states), the reference base description with that head's semantics
  None, as `model_capabilities` lists a head it has no words for.

And a test-only encoder declaration, never shipped as a product option
(`DoubleDeclaration`), declared for one test at a time by `declare_encoder`:
another output width and another sampling need than GATv2's, so a test shows
that no shared layer gates on `gatv2` or on layers == hops.

Module: tests/fixtures/capabilities.py
"""
from __future__ import annotations

from typing import Any

from src.models.bundle import spec as spec_module
from src.models.bundle.spec import FULL_VIEW, SAMPLED, EncoderDeclaration


class DoubleDeclaration(EncoderDeclaration):
    """A test-only encoder declaration, never shipped: its node vectors are
    `hidden` wide, its sampler takes two hops whatever its layers, it reads
    no relation feature and has no concat rule. No implementation exists."""

    kind = "double"
    computation = (FULL_VIEW, SAMPLED)
    hops_words = "one hop per sampled ring the double declares"

    def width(self, spec: Any) -> int:
        return spec.hidden

    def sampled_hops(self, spec: Any) -> int | None:
        return 2


def declare_encoder(monkeypatch: Any, declaration: EncoderDeclaration) -> None:
    """Declare `declaration`'s kind for one test (`monkeypatch` undoes it): a
    kind of the spec's list and a declaration of this build's, both read when
    a judgment is made."""
    kind = declaration.kind
    monkeypatch.setattr(spec_module, "ENCODER_KINDS", (*spec_module.ENCODER_KINDS, kind))
    monkeypatch.setattr(spec_module, "ENCODER_DECLARATIONS",
                        {**spec_module.ENCODER_DECLARATIONS, kind: declaration})


#: The `task` a record of `described` names for a base whose LP head declares
#: scoring semantics this build does not state (and no task head): a model no
#: protocol of this build takes.
LP_HEAD_UNSTATED = "lp_head_unstated"


def described(stage: str | None, task: str | None) -> dict[str, Any] | None:
    """The capability description of a record of `stage` and `task` (above)."""
    from src.models.bundle.capabilities import (
        LINK_PREDICTION,
        TASK,
        TASK_HEAD_SEMANTICS,
        model_capabilities,
    )
    from src.models.bundle.spec import TASKS, reference_spec

    if stage is None:
        return None
    if task == LP_HEAD_UNSTATED:
        base = model_capabilities(reference_spec("base", None))
        return {**base, "heads": tuple((kind, name, None) if kind == LINK_PREDICTION
                                       else (kind, name, semantics)
                                       for kind, name, semantics in base["heads"])}
    if (stage, task) == ("base", None) or (stage == "finetune" and task in TASKS):
        return model_capabilities(reference_spec(stage, task))
    nearest = model_capabilities(reference_spec(stage, "disease" if stage == "finetune" else None))
    heads = tuple(head for head in nearest["heads"] if head[0] != TASK)
    if task is not None:
        heads += ((TASK, task, TASK_HEAD_SEMANTICS.get(task, task)),)
    return {**nearest, "heads": heads}


__all__ = ["DoubleDeclaration", "LP_HEAD_UNSTATED", "declare_encoder", "described"]
