"""
The reference training settings of each learning stage, with their sources.
==========================================================================
`docs/working/scorer-retraining/README.md` §0.3, C6, U3, U8, U15. One place:
in a new run (a base, a finetune) `scripts/train_model.py` fills every setting
the operator did not give from here, and records which it filled; a resumed,
extended or branched run takes its parent's settings instead, recorded as
inherited. Every value is a provisional default
anchored on the fixed source (§0.3), never a claim that it is tuned for this
project's data; an operator's value replaces it and is recorded as theirs.

Model architecture is the model spec's (`src.models.bundle.spec`), not here.
Plain data, in `src.config` so the run planner (`src.utils.run_plan`) and the
training service read the same lists as training does.

Module: src/config/recipe.py
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

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
    "early_stopping_patience": None,
}

#: Early stopping is disabled in the source (C6): no patience. An operator who
#: sets one asks for early stopping, recorded as theirs; a continued run
#: inherits the patience in effect, so on or off, with the other settings.
_NO_EARLY_STOPPING = "early stopping disabled in the source (README C6): no patience"

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
    "early_stopping_patience": _NO_EARLY_STOPPING,
}

#: Settings that are not training choices: the base selects on LP accuracy
#: (A5), and its validation pairs depend on the split alone (every validation
#: edge once, ordered and given negatives by the split's seed, in batches of
#: this size), so runs over one split, whatever their seed or batch size,
#: validate on the same pairs (F3).
BASE_FIXED: Dict[str, Any] = {"monitor": "val_lp_accuracy", "monitor_mode": "max",
                              "validation_seed": "split_seed", "validation_batch_size": 512}


#: Finetune settings shared by the tasks, README U3, U8, U15, C6.
FINETUNE: Dict[str, Any] = {
    "num_epochs": 100,
    "weight_decay": 0.0,
    "batch_size": 64,
    "max_grad_norm": 1.0,
    "seed": 33,
    "scheduler_type": "none",
    "warmup_steps": 0,
    "use_amp": False,
    "early_stopping_patience": None,
}

FINETUNE_SOURCES: Dict[str, str] = {
    "num_epochs": f"{UPSTREAM} hparams.py:95-110 (max_epochs 100)",
    "weight_decay": f"{UPSTREAM} gene_prioritization_model.py:541-543 (Adam, no weight decay; U8)",
    "batch_size": f"{UPSTREAM} hparams.py:95-110 (64 patients)",
    "max_grad_norm": f"{UPSTREAM} hparams.py:95-110 (gradclip 1.0)",
    "seed": f"{UPSTREAM} train.py:78 (seed 33)",
    "scheduler_type": f"{UPSTREAM} train.py:283-292,311 (no scheduler; README C6, U8)",
    "warmup_steps": "no warmup in the source (README C6)",
    "use_amp": f"{UPSTREAM} train.py:308-317 (precision 16 never passed: fp32; README U15)",
    "early_stopping_patience": _NO_EARLY_STOPPING,
}

#: Per task: learning rate, first-hop size, sparse sample and the two named
#: loss weights (README U3, U8, C3: the scripts' lambda flipped into
#: task/LP weights).
FINETUNE_TASKS: Dict[str, Dict[str, Any]] = {
    "disease": {"learning_rate": 1e-5, "num_neighbors": [15, 10, 5],
                "sparse_sample_size": 300, "task_loss_weight": 0.1, "lp_loss_weight": 0.9,
                "candidate_sample_size": 1000},
    "gene": {"learning_rate": 1e-5, "num_neighbors": [-1, 10, 5],
             "sparse_sample_size": 100, "task_loss_weight": 0.6, "lp_loss_weight": 0.4,
             "candidate_upsample_size": 2},
    "patients_like_me": {"learning_rate": 5e-5, "num_neighbors": [15, 10, 5],
                         "sparse_sample_size": 300, "task_loss_weight": 0.3,
                         "lp_loss_weight": 0.7, "candidate_upsample_size": 1},
}

FINETUNE_TASK_SOURCES: Dict[str, Dict[str, str]] = {
    "disease": {
        "learning_rate": f"{UPSTREAM} disease run_*.sh script (lr 1e-5; README U3, U8)",
        "num_neighbors": f"{UPSTREAM} hparams.py:74 and the disease run_*.sh script (first hop "
                         "15, then 10, 5; README U3)",
        "sparse_sample_size": f"{UPSTREAM} disease run_*.sh script (sparse_sample 300; README U3)",
        "task_loss_weight": f"{UPSTREAM} disease run_*.sh script lmbda 0.9 on LP, so 0.1 on "
                            "the task (README C3)",
        "lp_loss_weight": f"{UPSTREAM} disease run_*.sh script lmbda 0.9 (README C3)",
        "candidate_sample_size": f"{UPSTREAM} hparams.py:101 (n_cand_diseases 1000 per batch)",
    },
    "gene": {
        "learning_rate": f"{UPSTREAM} gene run_*.sh script (lr 1e-5; README U3, U8)",
        "num_neighbors": f"{UPSTREAM} hparams.py:74 and the gene run_*.sh script (first hop "
                         "-1, all neighbours, then 10, 5; README U3)",
        "sparse_sample_size": f"{UPSTREAM} gene run_*.sh script (sparse_sample 100; README U3)",
        "task_loss_weight": f"{UPSTREAM} gene run_*.sh script lmbda 0.4 on LP, so 0.6 on the "
                            "task (README C3)",
        "lp_loss_weight": f"{UPSTREAM} gene run_*.sh script lmbda 0.4 (README C3)",
        "candidate_upsample_size": f"{UPSTREAM} gene run_*.sh script (upsample_cand 2; "
                                   "README U3, U14)",
    },
    "patients_like_me": {
        "learning_rate": f"{UPSTREAM} patients-like-me run_*.sh script (lr 5e-5; README U3, U8)",
        "num_neighbors": f"{UPSTREAM} hparams.py:74 and the patients-like-me run_*.sh script "
                         "(first hop 15, then 10, 5; README U3)",
        "sparse_sample_size": f"{UPSTREAM} patients-like-me run_*.sh script (sparse_sample 300; "
                              "README U3)",
        "task_loss_weight": f"{UPSTREAM} patients-like-me run_*.sh script lmbda 0.7 on LP, so "
                            "0.3 on the task (README C3)",
        "lp_loss_weight": f"{UPSTREAM} patients-like-me run_*.sh script lmbda 0.7 (README C3)",
        "candidate_upsample_size": f"{UPSTREAM} run_patients_like_me.sh (upsample_cand 1; "
                                   "README U3, U9, U14)",
    },
}

#: The finetune selects on pooled MRR (E1).
FINETUNE_FIXED: Dict[str, Any] = {"monitor": "val_pooled_mrr", "monitor_mode": "max"}

#: Settings a branch may change beside the stage's reference settings
#: (BACKLOG 23: the schedule's floor, the precision and the accumulation).
BRANCH_COMPANIONS = ("min_lr_ratio", "amp_dtype", "gradient_accumulation_steps")
#: Reference settings a branch keeps: early stopping on or off is not a
#: setting to compare runs by this round.
BRANCH_KEPT = ("early_stopping_patience",)


def reference_settings(learning_stage: str, task: Optional[str] = None
                       ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The reference settings and their sources for a stage (and task)."""
    if learning_stage == "base":
        if task is not None:
            raise ValueError("a base stage has no task")
        return dict(BASE), dict(BASE_SOURCES)
    if learning_stage == "finetune":
        if task not in FINETUNE_TASKS:
            raise ValueError(f"no reference finetune settings for task {task!r} in this build")
        values = {**FINETUNE, **FINETUNE_TASKS[task]}
        sources = {**FINETUNE_SOURCES, **FINETUNE_TASK_SOURCES[task]}
        return values, sources
    raise ValueError(f"no reference training settings for learning stage {learning_stage!r}")


def unread_settings(learning_stage: str, task: Optional[str] = None) -> FrozenSet[str]:
    """The recipe's settings that this stage and task do not read: another
    task's (`candidate_sample_size` on a gene run), or any task setting on a
    base. Set by the operator, they are refused, never ignored (README §7)."""
    every = set(BASE).union(FINETUNE, *FINETUNE_TASKS.values())
    values, _ = reference_settings(learning_stage, task)
    return frozenset(every - set(values))


def branch_settings(learning_stage: str, task: Optional[str] = None) -> FrozenSet[str]:
    """The closed list of settings a branch of this stage and task may change
    (README §1, BACKLOG 23): the reference settings, but `BRANCH_KEPT`, and
    `BRANCH_COMPANIONS`. Every other setting, the model spec, the split seed
    and the data stay the parent's."""
    values, _ = reference_settings(learning_stage, task)
    return frozenset(set(values) - set(BRANCH_KEPT) | set(BRANCH_COMPANIONS))


__all__ = ["BASE", "BASE_FIXED", "BASE_SOURCES", "BRANCH_COMPANIONS", "BRANCH_KEPT", "FINETUNE",
           "FINETUNE_FIXED", "FINETUNE_SOURCES", "FINETUNE_TASKS", "FINETUNE_TASK_SOURCES",
           "branch_settings", "reference_settings", "unread_settings"]
