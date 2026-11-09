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
  semantics is its name), and nothing else changed.

Module: tests/fixtures/capabilities.py
"""
from __future__ import annotations

from typing import Any


def described(stage: str | None, task: str | None) -> dict[str, Any] | None:
    """The capability description of a record of `stage` and `task` (above)."""
    from src.models.bundle.capabilities import TASK, TASK_HEAD_SEMANTICS, model_capabilities
    from src.models.bundle.spec import TASKS, reference_spec

    if stage is None:
        return None
    if (stage, task) == ("base", None) or (stage == "finetune" and task in TASKS):
        return model_capabilities(reference_spec(stage, task))
    nearest = model_capabilities(reference_spec(stage, "disease" if stage == "finetune" else None))
    heads = tuple(head for head in nearest["heads"] if head[0] != TASK)
    if task is not None:
        heads += ((TASK, task, TASK_HEAD_SEMANTICS.get(task, task)),)
    return {**nearest, "heads": heads}


__all__ = ["described"]
