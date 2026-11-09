#!/usr/bin/env python
"""
What contract M2 refuses in a directory of workspaces, before a deployment.
===========================================================================
**`--inventory` (M2.6).** Given a directory of workspaces (`data/workspaces`),
it lists, for each workspace, whether its graph and its training inputs pass the
consumers' checks, and for every checkpoint in it whether its training record
passes M2's check for serving and measurement (M2.4) and for a resume (M2.3),
and whether it carries the model record this build reads (a file without a model
spec is refused for both), each with the refusal's reason. The rules are the
consumers' own functions, called on reads made here, not restated. **Read-only**:
it moves, deletes and rewrites nothing; it writes only `--output`. What it lists
is archived by the owner, not deleted. Only names relative to the inventoried
directory are recorded.

What it does not judge: anything about a checkpoint beyond its training record
and its model record -- its learning stage or task, whether its weights build a
model, or a device has the capacity. Checkpoints are read once with
`read_checkpoint` and the safe loader, where serving uses the full one, so one
the safe loader cannot open is not judged.

**The M1-M3 mode retired with the files it read** (K8-7b). `--checkpoint-dir`
summarised one directory of checkpoints written before the model spec (the keys
they carry, their input width, whether a filename's score is their `val_mrr`),
the facts that established the frozen evaluator could not be the calibration
oracle (backlog item 10). Its evidence files stay as they were recorded
(`docs/working/EVIDENCE_M1_M3_gat.json`, `EVIDENCE_M1_M3_hgt.json`); this build
ranks no checkpoint by `val_mrr` and writes none without a model spec, so the
mode has nothing of this build's left to read.

**Aggregates and relative names only.** BACKLOG §5.2 fixes what these files may
contain, because they describe a clinical deployment: no tensors, no absolute
paths, no operator or host names.

Usage:
    python scripts/audit_checkpoint_family.py \\
        --inventory data/workspaces --output <outside the repository>/inventory.json

Module: scripts/audit_checkpoint_family.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shared with the other evidence scripts rather than restated here: all three
# reports are read together, and a claim spelled differently in each cannot be
# compared across them.
from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

logger = logging.getLogger(__name__)


def _runtime_facts() -> Dict[str, Any]:
    """The **reader's** environment, without saying where it is or whose it is.

    Not a machine identity: architecture, OS, Python, torch, CUDA and the GPU
    *model* name. No hostname, no serial, no device uuid — §5.2 forbids them and
    none of them would add anything, since nothing here is a join key.

    What these facts are for is narrower than "provenance". The inventory opens
    checkpoints with `weights_only=True`, and whether a given checkpoint loads
    that way is a property of the torch version doing the opening, so a "not
    judged" verdict is only interpretable next to `torch`. The rest is context
    of the same kind. They say nothing about whether this machine is the
    deployment's; that claim is `--deployment-relationship`, made by a person,
    the same split `MeasurementManifest.cuda_executed` already uses.
    """
    import platform

    import torch

    facts = {
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": None,
    }
    if facts["cuda_available"]:
        facts["device_name"] = torch.cuda.get_device_name(0)
    return facts


# =============================================================================
# --inventory: what contract M2 refuses (M2.6)
# =============================================================================
#: Files at a workspace's root that are not checkpoints: the graph tensors, and
#: `shortest_paths.pt`, a table an earlier build may have left there. Nothing in
#: this build reads that table; its name is here only so that a leftover file,
#: which can be gigabytes, is never read whole and listed as a checkpoint.
_NOT_CHECKPOINTS = ("node_features.pt", "edge_indices.pt", "shortest_paths.pt")
#: The split artifact `scripts/train_model.py` writes in each run directory
#: (F2), beside the run's checkpoints: not a checkpoint.
_RUN_SPLIT = "graph_split.pt"


def _scrub(text: str, root: Path) -> str:
    """A refusal's message with the inventoried directory's location removed."""
    for form in sorted({str(root.resolve()), str(root)}, key=len, reverse=True):
        text = text.replace(form + "/", "").replace(form, ".")
    return text


def _judge(check: Any, root: Path) -> Dict[str, Any]:
    """Run one consumer check: accepted, or refused with its reason."""
    try:
        check()
    except Exception as exc:  # noqa: BLE001 - a refusal is the finding
        return {"verdict": "refused",
                "reason": f"{type(exc).__name__}: {_scrub(str(exc), root)}"}
    return {"verdict": "accepted", "reason": None}


def _not_judged(reason: str) -> Dict[str, Any]:
    return {"verdict": "not judged", "reason": reason}


def _graph_check(workspace: Path) -> Tuple[Any, Dict[str, str]]:
    """The graph check serving and measurement make, from one read of each file:
    the tensors against the manifest, `kg.json` identified against it. Returns
    the manifest reading and the manifest-bound digests M2.4 compares a
    checkpoint with."""
    from src.kg.artifacts import check_unparsed_kg_json, read_split_manifest, verify_graph_reads
    from src.kg.storage.file_storage import read_graph_artifacts

    manifest = read_split_manifest(workspace)
    graph = read_graph_artifacts(workspace, map_location="cpu")
    bound = verify_graph_reads(manifest, graph.reads)
    check_unparsed_kg_json(workspace, manifest)
    return manifest, dict(bound)


def _serving_check(workspace: Path) -> Dict[str, str]:
    """Every workspace check serving and measurement make before a checkpoint is
    read: the graph, then the dataset record (contract M3b). Returns the
    manifest-bound digests."""
    from src.kg.artifacts import verify_dataset_record

    manifest, bound = _graph_check(workspace)
    verify_dataset_record(manifest, bound["kg"])
    return bound


def _dataset_record(manifest: Any, kg_digest: str, root: Path) -> Dict[str, Any]:
    """The dataset record's verdict (contract M3b), with its binding state and its
    origin as separate fields, so a synthetic workspace is never listed as
    plainly verified and a refusal names the state that refused it."""
    from src.kg.artifacts import DatasetRecordRefused, verify_dataset_record

    try:
        record = verify_dataset_record(manifest, kg_digest)
    except DatasetRecordRefused as exc:
        return {"verdict": "refused", "reason": f"{type(exc).__name__}: {_scrub(str(exc), root)}",
                "binding": exc.binding, "origin": exc.origin}
    except Exception as exc:  # noqa: BLE001 - one workspace never stops the listing
        return {"verdict": "refused", "reason": f"{type(exc).__name__}: {_scrub(str(exc), root)}",
                "binding": None, "origin": None}
    summary = record.summary()
    return {"verdict": "accepted", "reason": None, "binding": summary["binding"],
            "origin": summary["origin"], "sha256": summary["sha256"],
            "incomplete_by_design": summary["incomplete_by_design"]}


#: What a checkpoint's verdicts cover, stated in the report beside them.
CHECKPOINT_SCOPE = (
    "M2's source-contract checks and the model record this build reads: whether the "
    "checkpoint's training record names this workspace's graph (serve_and_measure, "
    "M2.4) and this workspace's training inputs (resume, M2.3), and whether it carries "
    "a model spec (a file without one is refused for both, as serving, the task "
    "measurements and training refuse it). The serve_and_measure verdict follows "
    "serving and the task measurements (since K8-6a no command measures a checkpoint "
    "without a model spec). Not its learning stage or task (a base reads accepted for "
    "serve_and_measure, although serving refuses it), whether its weights build a "
    "model over the graph, whether a device has the capacity, or whether the service "
    "runs."
)


def _checkpoint_verdicts(
    path: Path, root: Path, bound: Optional[Dict[str, str]], inputs: Any,
    workspace_refused: str = "the workspace's graph is refused",
) -> Dict[str, Any]:
    """One checkpoint, read once (`read_checkpoint`, the safe loader on CPU),
    judged from that read; its digest is that read's.

    Each use is judged in its consumer's order: serving checks the training
    record of the graph, then reads the model record (`read_model_record`); a
    resume reads the parent's model record, then compares its training inputs.
    A file without a model spec is refused by `read_model_record`'s own reason."""
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.models.bundle.record import read_model_record
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.fingerprint import verify_training_record

    name = path.relative_to(root).as_posix()
    try:
        read = read_checkpoint(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - an unreadable checkpoint is data
        why = f"the safe loader cannot open it ({type(exc).__name__})"
        return {"sha256": None,
                "m2_contract": {"serve_and_measure": _not_judged(why),
                                "resume": _not_judged(why)}}
    checkpoint = read.checkpoint

    if bound is None:
        serve = _not_judged(workspace_refused)
    else:
        serve = _judge(lambda: (verify_training_record(
            checkpoint, bound, tuple(GRAPH_ARTIFACTS),
            recorded_by=f"checkpoint {name}", compared_with="this workspace's graph",
        ), read_model_record(checkpoint)), root)
    # A model bundle checkpoint is judged against the read its own stage and
    # task make (`inputs` is then a function of the checkpoint), as training
    # judges it; anything else against the workspace's one read.
    own = inputs(checkpoint) if callable(inputs) else inputs
    if own is None:
        resume = _not_judged("training refuses this workspace")
    elif isinstance(own, Exception):
        resume = _not_judged(f"training refuses this workspace for this checkpoint's stage "
                             f"and task ({type(own).__name__}: {_scrub(str(own), root)})")
    else:
        from scripts.train_model import check_resume_parent

        resume = _judge(lambda: (read_model_record(checkpoint), check_resume_parent(
            checkpoint, own, recorded_by=f"checkpoint {name}",
            compared_with="a run over this workspace",
        )), root)
    return {"sha256": read.identity.sha256,
            "m2_contract": {"serve_and_measure": serve, "resume": resume}}


def inventory_workspace(workspace: Path, root: Path) -> Dict[str, Any]:
    """One workspace: its graph, its dataset record, its training inputs, its
    checkpoints."""
    from scripts.train_model import read_training_inputs

    found: Dict[str, Any] = {}

    def graph() -> None:
        found["manifest"], found["bound"] = _graph_check(workspace)

    def training() -> None:
        found["inputs"] = read_training_inputs(workspace)

    entry: Dict[str, Any] = {"graph": _judge(graph, root)}
    bound = found.get("bound")
    if bound is None:
        entry["dataset_record"] = {**_not_judged("the workspace's graph is refused"),
                                   "binding": None, "origin": None}
    else:
        entry["dataset_record"] = _dataset_record(found["manifest"], bound["kg"], root)
    entry["training"] = _judge(training, root)
    inputs = found.get("inputs")
    # Serving and measurement refuse the workspace when either check does
    # (contract M3b), so no checkpoint is judged servable over a refused record.
    served_bound, refused = bound, "the workspace's graph is refused"
    if bound is not None and entry["dataset_record"]["verdict"] != "accepted":
        served_bound, refused = None, "the workspace's dataset record is refused"
    checkpoints = sorted(
        path for path in workspace.rglob("*.pt")
        if not (path.parent == workspace and path.name in _NOT_CHECKPOINTS)
        and path.name != _RUN_SPLIT
    )
    reads: Dict[Any, Any] = {}

    def inputs_for(checkpoint: Any) -> Any:
        """The training inputs a run continuing `checkpoint` reads: by its
        model record's stage and task for a model bundle (a base reads the
        graph only, a gene or patients-like-me child the fixture too), else the
        workspace's one read. One read per (stage, task)."""
        from src.models.bundle.record import ModelRecordError, has_model_spec, read_model_record

        if not isinstance(checkpoint, dict) or not has_model_spec(checkpoint):
            return inputs
        try:
            record = read_model_record(checkpoint)
        except ModelRecordError as exc:
            return exc
        key = (record.learning_stage, record.task)
        if key not in reads:
            try:
                reads[key] = read_training_inputs(
                    workspace, with_samples=key[0] == "finetune",
                    task=key[1] if key[0] == "finetune" else None)
            except Exception as exc:  # noqa: BLE001 - a refusal is the verdict
                reads[key] = exc
        return reads[key]

    entry["checkpoints"] = {
        path.relative_to(root).as_posix(): _checkpoint_verdicts(
            path, root, served_bound, inputs_for, refused)
        for path in checkpoints
    }
    return entry


def build_inventory(root: Path, relationship: str) -> Dict[str, Any]:
    workspaces = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspaces:
        raise SystemExit(f"{root} holds no workspace directories")
    listing = {path.name: inventory_workspace(path, root) for path in workspaces}

    counts: Counter = Counter()
    for entry in listing.values():
        for part in ("graph", "dataset_record", "training"):
            counts[f"{part}:{entry[part]['verdict']}"] += 1
        if entry["dataset_record"]["origin"] is not None:
            counts[f"dataset_record origin:{entry['dataset_record']['origin']}"] += 1
        for verdicts in entry["checkpoints"].values():
            for use, verdict in verdicts["m2_contract"].items():
                counts[f"checkpoint m2_contract {use}:{verdict['verdict']}"] += 1
    return {
        "fact": "M2.6 inventory",
        "what_this_shows": (
            "for each workspace, whether contract M2 accepts its graph, its training "
            "inputs and each checkpoint's training record, whether each checkpoint "
            "carries the model record this build reads, and whether contract M3b "
            "accepts its dataset record (binding state and origin apart), and why not"
        ),
        "read_only": (
            "nothing was moved, deleted or rewritten; refused artifacts are archived "
            "by the owner, not deleted"
        ),
        "checkpoint_verdicts_cover": CHECKPOINT_SCOPE,
        "not_checked": [
            "checkpoints the safe loader cannot open (listed as not judged)",
        ],
        "counts": dict(sorted(counts.items())),
        "workspaces": listing,
        "runtime": _runtime_facts(),
        "deployment_relationship": relationship,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="What contract M2 refuses in a directory of workspaces (M2.6)")
    parser.add_argument("--inventory", type=Path, required=True,
                        help="A directory of workspaces: list what contract M2 refuses "
                             "in each, and why (M2.6). Read-only.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where the evidence JSON is written")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing --output. Off by default: evidence "
                             "artifacts are cited by digest and must not be replaced "
                             "silently.")
    parser.add_argument("--deployment-relationship", default=UNSTATED_RELATIONSHIP,
                        choices=DEPLOYMENT_RELATIONSHIPS,
                        help="How this machine relates to the deployment. A bounded "
                             "vocabulary rather than free text: the schema forbids "
                             "operator and host names, and cannot then accept an "
                             "arbitrary string. Unverified by design.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    subject = args.inventory
    if not subject.is_dir():
        raise SystemExit(f"not a directory: {subject}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.output} exists. Pass --overwrite to replace it, or write "
            "elsewhere — an evidence file that was silently replaced is not evidence."
        )

    report = build_inventory(args.inventory, args.deployment_relationship)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    logger.info("%s -> %s", report["counts"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
