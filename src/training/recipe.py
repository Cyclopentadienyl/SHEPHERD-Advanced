"""
The reference training settings of each learning stage, with their sources.
==========================================================================
`docs/working/scorer-retraining/README.md` §0.3, C6, U3, U8, U15. One place:
`scripts/train_model.py` fills every setting the operator did not give from
here, and records which it filled. Every value is a provisional default
anchored on the fixed source (§0.3), never a claim that it is tuned for this
project's data; an operator's value replaces it and is recorded as theirs.

Model architecture is the model spec's (`src.models.bundle.spec`), not here.

Module: src/training/recipe.py
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

UPSTREAM = "mims-harvard/SHEPHERD@e95433a"

#: Base (KG link prediction) settings, README U3, U15, C6.
BASE: Dict[str, Any] = {
    "num_epochs": 200,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "batch_size": 512,
    "num_neighbors": [15, 10, 5],
    "max_grad_norm": 1.0,
    "seed": 1,
    "scheduler_type": "none",
    "warmup_steps": 0,
    "use_amp": False,
}

BASE_SOURCES: Dict[str, str] = {
    "num_epochs": f"{UPSTREAM} hparams.py:18-44 (200 epochs)",
    "learning_rate": f"{UPSTREAM} run_pretrain.sh:6-17 (lr 1e-4)",
    "weight_decay": f"{UPSTREAM} run_pretrain.sh:6-17 (wd 0)",
    "batch_size": f"{UPSTREAM} hparams.py:18-44 (512 sources a batch)",
    "num_neighbors": f"{UPSTREAM} hparams.py:18-44 (neighbour sizes [15, 10, 5])",
    "max_grad_norm": f"{UPSTREAM} hparams.py:18-44 (gradient clip 1.0)",
    "seed": f"{UPSTREAM} hparams.py:18-44 (seed 1)",
    "scheduler_type": f"{UPSTREAM} pretrain.py:97,109-117 (ReduceLROnPlateau patience 1000 "
                      "never fires within 200 epochs: a constant rate; README C6)",
    "warmup_steps": "no warmup in the source (README C6)",
    "use_amp": f"{UPSTREAM} pretrain.py:109-117 (precision 16 never passed: fp32; README U15)",
}

#: Settings that are not training choices of the recipe: early stopping is
#: disabled (C6), and the base selects on LP accuracy (A5).
BASE_FIXED: Dict[str, Any] = {"early_stopping": False, "monitor": "val_lp_accuracy",
                              "monitor_mode": "max"}


def reference_settings(learning_stage: str, task: Optional[str] = None
                       ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The reference settings and their sources for a stage (and task)."""
    if learning_stage == "base":
        if task is not None:
            raise ValueError("a base stage has no task")
        return dict(BASE), dict(BASE_SOURCES)
    raise ValueError(f"no reference training settings for learning stage {learning_stage!r} "
                     "in this build")


__all__ = ["BASE", "BASE_FIXED", "BASE_SOURCES", "reference_settings"]
