"""A synthetic model family world, for the family graph's tests (Models' By family).

Checkpoint files whose model records the record writer builds
(`test_task_first_selection._record_payload`, the record a training payload
writes), filed where the server's list finds them, so the graph is drawn from
what the one listing and its family view read (`server_lists.list_family`),
never from a test's own rows. What it holds (G/I of the stage-lineage
package):

- **family alpha**: a base run with two saved files; a disease fine-tune from
  one file and a gene fine-tune and a frozen disease fine-tune from the other
  (two children from two files of one parent run); a branch of the base; a
  chain of extends under the disease fine-tune five runs deep, so the deepest
  run sits seven levels from the root;
- **family beta**: a base with three disease fine-tunes, two of them named
  alike (told apart by their start time) and one with a long name;
- **a parent elsewhere**: a fine-tune whose parent file is in another listed
  workspace (``other``);
- **a parent unavailable**: a fine-tune whose recorded parent run is named
  with markup (`EVIL`) and listed nowhere;
- **a refused record**: a file with no model spec.

Module: tests/fixtures/family_world.py
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: A recorded parent run named with markup: drawn as text, never as HTML.
EVIL = "<img src=x onerror=alert(1)>"

#: Each run: (label, run directory, files, parent (label, file) or a recorded
#: parent, task, operation, family root label, weight transfer). Listed newest
#: first by file time: later rows are older.
_FROZEN = {"parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                     "task_head": "trainable"}}
_FULL = {"parts": {"input": "trainable", "encoder": "trainable", "lp_head": "trainable",
                   "task_head": "trainable"}}

RUNS: Tuple[Tuple[Any, ...], ...] = (
    ("alpha-base", "20261101-000000_alpha-base_a00001", ("last.pt", "model-03.pt"), None, None,
     "new", "alpha-base", None),
    ("alpha-disease", "20261102-000000_alpha-disease_a00002", ("last.pt",),
     ("alpha-base", "model-03.pt"), "disease", "finetune", "alpha-base", _FULL),
    ("alpha-gene", "20261103-000000_alpha-gene_a00003", ("last.pt",),
     ("alpha-base", "last.pt"), "gene", "finetune", "alpha-base", _FULL),
    ("alpha-frozen", "20261104-000000_alpha-frozen_a00004", ("last.pt",),
     ("alpha-base", "last.pt"), "disease", "finetune", "alpha-base", _FROZEN),
    ("alpha-branch", "20261105-000000_alpha-branch_a00005", ("last.pt",),
     ("alpha-base", "last.pt"), None, "branch", "alpha-base", _FULL),
    ("alpha-ext-1", "20261106-000000_alpha-ext-1_a00006", ("last.pt",),
     ("alpha-disease", "last.pt"), "disease", "extend", "alpha-base", None),
    ("alpha-ext-2", "20261107-000000_alpha-ext-2_a00007", ("last.pt",),
     ("alpha-ext-1", "last.pt"), "disease", "extend", "alpha-base", None),
    ("alpha-ext-3", "20261108-000000_alpha-ext-3_a00008", ("last.pt",),
     ("alpha-ext-2", "last.pt"), "disease", "extend", "alpha-base", None),
    ("alpha-ext-4", "20261109-000000_alpha-ext-4_a00009", ("last.pt",),
     ("alpha-ext-3", "last.pt"), "disease", "extend", "alpha-base", None),
    ("alpha-ext-5", "20261110-000000_alpha-ext-5_a00010", ("last.pt",),
     ("alpha-ext-4", "last.pt"), "disease", "extend", "alpha-base", None),
    ("beta-base", "20261111-000000_beta-base_b00001", ("last.pt",), None, None, "new",
     "beta-base", None),
    ("beta-disease-1", "20261112-080000_beta-disease_b00002", ("last.pt",),
     ("beta-base", "last.pt"), "disease", "finetune", "beta-base", _FULL),
    ("beta-disease-2", "20261112-090000_beta-disease_b00003", ("last.pt",),
     ("beta-base", "last.pt"), "disease", "finetune", "beta-base", _FULL),
    ("beta-long", "20261113-000000_beta-disease-with-a-rather-long-descriptive-run-name_b00004",
     ("last.pt",), ("beta-base", "last.pt"), "disease", "finetune", "beta-base", _FULL),
    ("gamma-disease", "20261114-000000_gamma-disease_c00002", ("last.pt",), "elsewhere",
     "disease", "finetune", "gamma-base", _FULL),
    ("delta-disease", "20261115-000000_delta-disease_d00002", ("last.pt",),
     {"sha256": "9" * 64, "run": EVIL}, "disease", "finetune", "delta-base", _FULL),
)
_RUN_DIRS = {row[0]: row[1] for row in RUNS}
#: A file with no model spec: its record is refused, listed with why.
REFUSED_RUN = "20261116-000000_old-gnn_e00001"
#: The parent the gamma fine-tune names, in the other workspace.
OTHER_BASE_RUN = "20261020-000000_gamma-base_c00001"


def run_key(label: str) -> str:
    """A run's identity in the listing (``architecture/run``)."""
    return f"gatv2/{_RUN_DIRS[label]}"


def file_key(label: str, name: str = "last.pt") -> str:
    """A saved file's identity in the listing (``architecture/run/file``)."""
    return f"{run_key(label)}/{name}"


def _put(workspace: Path, run: str, name: str, payload: Dict[str, Any], age: float) -> str:
    from src.utils.checkpoint_io import write_checkpoint

    path = workspace / "runs" / "gatv2" / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = write_checkpoint(payload, path)
    stamp = time.time() - age * 3600
    os.utime(path, (stamp, stamp))
    return sha


def write_family_world(root: Path, demo: Path, *, other: Optional[str] = "other"
                       ) -> Dict[str, Any]:
    """Workspaces under `root` (``demo``, and `other` holding the gamma
    fine-tune's parent unless None), each a copy of the demo workspace
    `demo`, with the runs of `RUNS` and `REFUSED_RUN` in ``demo``. Returns
    ``{"workspace", "other", "shas"}``, ``shas`` by ``(label, file)``."""
    from src.utils.fingerprint import file_sha256
    from tests.unit import test_task_first_selection as bundles

    workspace = shutil.copytree(demo, root / "demo")
    fingerprint = file_sha256(workspace / "split_manifest.json")
    meta = {"checkpoint_kind": "weights", "training_input_digests": {"split_manifest": fingerprint}}
    shas: Dict[Tuple[str, str], str] = {}
    other_dir = None
    if other is not None:
        other_dir = shutil.copytree(demo, root / other)
        shas[("gamma-base", "last.pt")] = _put(other_dir, OTHER_BASE_RUN, "last.pt", {
            **bundles._record_payload(logs={"val_lp_accuracy": 0.7}, family_id="gamma-base"),
            **meta, "epoch": 4}, 40)
    for age, (label, run, files, parent, task, operation, family, transfer) in enumerate(
            RUNS):
        for index, name in enumerate(files):
            if parent is None:
                payload = bundles._record_payload(logs={"val_lp_accuracy": 0.8 - index / 10},
                                                  family_id=run)
            else:
                if parent == "elsewhere":
                    named = {"sha256": shas.get(("gamma-base", "last.pt"), "8" * 64),
                             "run": OTHER_BASE_RUN}
                elif isinstance(parent, dict):
                    named = dict(parent)
                else:
                    plabel, pfile = parent
                    named = {"sha256": shas[(plabel, pfile)],
                             "run": _RUN_DIRS[plabel]}
                root_run = _RUN_DIRS.get(family, family)
                payload = bundles._record_payload(task, logs={"val_pooled_mrr": 0.4},
                                                  parent=named, family_id=root_run)
                payload["operation"] = operation
                payload["weight_transfer"] = transfer
                if operation == "branch":
                    payload["parent"] = {**named, "changes": {}}
                if task is None:
                    # A base-stage branch: its record is a base's, from its parent.
                    payload["learning_stage"], payload["task"] = "base", None
            shas[(label, name)] = _put(workspace, run, name,
                                       {**payload, **meta, "epoch": 9 - 3 * index},
                                       len(RUNS) - age + index / 10)
    from tests.unit.test_task_first_selection import GNN_PAYLOAD

    shas[("old-gnn", "last.pt")] = _put(workspace, REFUSED_RUN, "last.pt", dict(GNN_PAYLOAD), 0)
    return {"workspace": workspace, "other": other_dir, "shas": shas}


__all__ = ["EVIL", "OTHER_BASE_RUN", "REFUSED_RUN", "RUNS", "file_key", "run_key",
           "write_family_world"]
