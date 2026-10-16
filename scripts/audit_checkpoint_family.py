#!/usr/bin/env python
"""
M1-M3 evidence — what a family of checkpoints actually carries.
===============================================================
Backlog item 10. Three facts were established by reading checkpoints in a review
thread and have existed only as pasted text since:

  M1  no scanned checkpoint carries `metadata` or `in_channels_dict`
  M2  the input width is 128, where the frozen evaluator's hardcoded fallback
      was 256 (`scripts/evaluate_model.py:144-148` at `7dab728`) — the size
      mismatch it died on
  M3  the number in a filename is `val_mrr`

Those three are what established that the frozen evaluator cannot be the
calibration oracle, which is the largest decision this phase has made. They are
also the ones a reviewer currently has to take on trust. This script makes them
reproducible from an artifact rather than from a summary.

**Aggregates only, and the exclusions are the point.** BACKLOG §5.2 fixes what
these files may contain, because they describe a clinical deployment and are
committed to the repository. This one records input digests, the checkpoint
count, a key-presence summary, an `in_channels` summary and the
filename-versus-`logs` comparison. It records **no tensors, no absolute paths and
no operator or host names** — only the basename of each checkpoint, which is the
subject of M3 and carries nothing else.

**A checkpoint that will not load is a finding, not a crash.** M1 exists because
the frozen evaluator's loader failed on this family, so a scan that stopped at the
first unreadable file would destroy the evidence it was run to collect.

**`--inventory`: what contract M2 refuses, before a deployment (M2.6).** Given a
directory of workspaces (`data/workspaces`), it lists, for each workspace, whether
its graph and its training inputs pass the consumers' checks, whether its
shortest-path table's source binding does, and for every checkpoint in it whether
its training record passes M2's check for serving and measurement (M2.4) and for
a resume (M2.3), each with the refusal's reason. The rules are the consumers' own
functions, called on reads made here, not restated. **Read-only**: it moves,
deletes and rewrites nothing; it writes only `--output`. What it lists is
archived by the owner, not deleted. Only names relative to the inventoried
directory are recorded.

What it does not judge: whether a table would be served (the tensor is not
loaded, since it can be gigabytes; a refused binding is still reported as a
refusal), and anything about a checkpoint beyond its training record -- whether
its weights build a model, or a device has the capacity. Checkpoints are read
once with `read_checkpoint` and the safe loader, as M1-M3 opens them, where
serving uses the full one, so one the safe loader cannot open is not judged.

Usage:
    python scripts/audit_checkpoint_family.py \\
        --checkpoint-dir data/workspaces/<ws>/runs/hgt/<run>/checkpoints \\
        --output docs/working/EVIDENCE_M1_M3_checkpoints.json

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

#: The keys whose presence M1 is about, plus the ones it observed instead.
KEYS_OF_INTEREST = (
    "metadata",
    "in_channels_dict",
    "config",
    "data_fingerprint",
    "training_input_digests",
    "epoch",
    "logs",
    "state_dict",
    "model_state_dict",
)

#: `feature_encoder.projections.<node_type>.weight` has shape
#: `(hidden_dim, in_channels)`, so its last dimension is the input width M2 is
#: about. Read from the weight rather than from a config field, because a config
#: records what was asked for and this records what was built.
_PROJECTION_PREFIX = "feature_encoder.projections."
_PROJECTION_SUFFIX = ".weight"


def filename_score_token(name: str) -> Optional[str]:
    """The score baked into a checkpoint filename, **as written**, or `None`.

    `model-45-0.6975.pt` gives `"0.6975"`. Returned as text because the text is the
    evidence: `float("0.7000")` is `0.7`, and `repr` of that recovers one decimal
    place rather than four, so a precision taken from the parsed value would widen
    the comparison tolerance by three orders of magnitude.

    **A decimal point is required.** `model-45.pt` carries an epoch and no score,
    and an earlier version read `45` as one — then compared it against a metric in
    [0, 1] and recorded a disagreement that was really a filename with nothing in
    it to compare.
    """
    stem = name[: -len(".pt")] if name.endswith(".pt") else name
    for token in reversed(stem.split("-")):
        if "." not in token:
            continue
        try:
            float(token)
        except ValueError:
            continue
        return token
    return None


def scores_agree(token: str, value: float) -> bool:
    """Whether a filename's score and a logs metric are the same number.

    **The rounding rule, stated rather than implied.** A filename carries a
    rendering of the metric rounded to however many decimals it was written with,
    so the metric is rounded the same way and the two are compared exactly. An
    earlier version compared with a tolerance derived from the parsed float, which
    was both too wide and wrong for trailing zeroes.
    """
    decimals = len(token.split(".")[1])
    return f"{round(value, decimals):.{decimals}f}" == f"{float(token):.{decimals}f}"


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    """One checkpoint, reduced to the aggregate facts M1-M3 are about."""
    import torch

    from src.utils.checkpoint_paths import ranking_score_detail

    record: Dict[str, Any] = {
        "filename": path.name,
        "filename_score_token": filename_score_token(path.name),
        "loaded": False,
        "load_error": None,
        "keys_present": [],
        "in_channels": {},
        "logs_metric": None,
        "logs_value": None,
    }

    try:
        # `weights_only=True` — the safe loader. A checkpoint this cannot read is
        # itself worth recording; falling back to executing pickled code to widen
        # the evidence would be the wrong trade in a clinical repository.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - an unreadable checkpoint is data
        # **The class, never the message.** Torch and filesystem errors routinely
        # interpolate the absolute path they failed on, and §5.2 forbids absolute
        # paths in this file. A stable category says as much as a reader here needs
        # and cannot carry a path, a username or a mount point out with it.
        record["load_error"] = type(exc).__name__
        return record

    if not isinstance(checkpoint, dict):
        record["load_error"] = "NotADict"
        return record

    record["loaded"] = True
    record["keys_present"] = sorted(k for k in KEYS_OF_INTEREST if k in checkpoint)

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or {}
    for key, tensor in state_dict.items():
        if key.startswith(_PROJECTION_PREFIX) and key.endswith(_PROJECTION_SUFFIX):
            node_type = key[len(_PROJECTION_PREFIX):-len(_PROJECTION_SUFFIX)]
            if hasattr(tensor, "shape") and len(tensor.shape) >= 1:
                record["in_channels"][node_type] = int(tensor.shape[-1])

    detail = ranking_score_detail(checkpoint.get("logs"))
    if detail is not None:
        record["logs_metric"], record["logs_value"] = detail[0], detail[1]
    return record


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The aggregate the three facts are read from.

    **Aggregates and filenames, no per-checkpoint block.** BACKLOG §5.2 asks for a
    key-presence *summary*, an `in_channels` *summary* and the filename-versus-logs
    comparison, and an evidence file committed to a clinical repository should not
    carry more than it was specified to. Filenames appear only where a count alone
    would strand a reader — which checkpoints were unreadable, which disagreed —
    and they are already `checkpoint_digests` keys, so no new category leaks.
    """
    loaded = [r for r in records if r["loaded"]]

    key_presence = Counter()
    for record in loaded:
        key_presence.update(record["keys_present"])

    widths = Counter()
    n_with_widths = 0
    for record in loaded:
        if record["in_channels"]:
            n_with_widths += 1
            widths.update(record["in_channels"].values())

    metrics = Counter(r["logs_metric"] for r in loaded if r["logs_metric"])

    agree, disagree, uncomparable = [], [], []
    for record in loaded:
        token, value = record["filename_score_token"], record["logs_value"]
        if token is None or value is None:
            uncomparable.append(record["filename"])
        elif scores_agree(token, value):
            agree.append(record["filename"])
        else:
            disagree.append(record["filename"])

    return {
        "n_checkpoints_found": len(records),
        "n_loaded": len(loaded),
        "n_unreadable": len(records) - len(loaded),
        "unreadable_filenames": sorted(r["filename"] for r in records if not r["loaded"]),
        "load_error_categories": dict(sorted(
            Counter(r["load_error"] for r in records if r["load_error"]).items())),
        "key_presence_counts": dict(sorted(key_presence.items())),
        # **How many checkpoints could answer M2 at all**, beside the answer. A
        # family whose projection weights are named differently would otherwise
        # produce an empty summary that reads like a normal result.
        "in_channels": {
            "n_loaded_exposing_projection_widths": n_with_widths,
            "n_loaded_without_projection_widths": len(loaded) - n_with_widths,
            "value_counts": {str(k): v for k, v in sorted(widths.items())},
            "established": bool(widths),
        },
        "logs_ranking_metric_counts": dict(sorted(metrics.items())),
        "filename_vs_logs": {
            "agree": len(agree),
            "disagree": len(disagree),
            "uncomparable": len(uncomparable),
            "disagreeing_filenames": sorted(disagree),
            "uncomparable_filenames": sorted(uncomparable),
        },
    }


def _runtime_facts() -> Dict[str, Any]:
    """The **reader's** environment, without saying where it is or whose it is.

    Not a machine identity: architecture, OS, Python, torch, CUDA and the GPU
    *model* name. No hostname, no serial, no device uuid — §5.2 forbids them and
    none of them would add anything, since nothing here is a join key.

    What these facts are for is narrower than "provenance". M1-M3 opens
    checkpoints with `weights_only=True`, and whether a given checkpoint loads
    that way is a property of the torch version doing the opening — so `n_loaded`
    is only interpretable next to `torch`. The rest is context of the same kind.
    They say nothing about whether this machine is the deployment's; that claim is
    `--deployment-relationship`, made by a person, the same split
    `MeasurementManifest.cuda_executed` already uses.
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


def build_report(checkpoint_dir: Path, relationship: str) -> Dict[str, Any]:
    from src.utils.fingerprint import file_sha256

    paths = sorted(checkpoint_dir.glob("*.pt"))
    if not paths:
        raise SystemExit(
            f"{checkpoint_dir} holds no *.pt files. An evidence file reporting zero "
            "checkpoints would read as a finding about the family rather than about "
            "the directory that was pointed at."
        )

    records = [inspect_checkpoint(path) for path in paths]
    if not any(r["loaded"] for r in records):
        raise SystemExit(
            f"none of the {len(paths)} checkpoints in {checkpoint_dir.name} could be "
            "loaded. Partial failure is evidence — M1 exists because a loader fails "
            "on this family — but total failure says nothing about what checkpoints "
            "carry, only that this reader could not open any of them."
        )

    return {
        "fact": "M1-M3",
        "what_this_shows": (
            "which keys a family of checkpoints carries, the input width they were "
            "built with, and whether the number in a filename is the ranking metric "
            "in its logs"
        ),
        "checkpoint_digests": {path.name: file_sha256(path) for path in paths},
        "summary": summarise(records),
        "runtime": _runtime_facts(),
        "deployment_relationship": relationship,
        "excluded_by_design": [
            "checkpoint tensors",
            "absolute paths (only basenames are recorded)",
            "operator and host names",
        ],
    }


# =============================================================================
# --inventory: what contract M2 refuses (M2.6)
# =============================================================================
#: Files at a workspace's root that are graph or shortest-path artifacts, not
#: checkpoints.
_NOT_CHECKPOINTS = ("node_features.pt", "edge_indices.pt", "shortest_paths.pt")


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


def _shortest_paths(workspace: Path, kg_digest: Optional[str], root: Path) -> Dict[str, Any]:
    """The table's source binding, from its sidecar (M2.5), and what that allows
    to be said about whether it would be served.

    **Two verdicts, because the inventory reads half of the artifact.** The
    source binding is judged from the sidecar with the loader's own functions.
    Whether the table would be served also depends on the tensor -- its columns
    and values, its build id against the sidecar's, and the hop bound, which the
    loader checks only for a table with rows -- and the tensor is not loaded
    here (it can be gigabytes). So `served` is `refused` where the binding
    already refuses it, and otherwise `not judged`; never `accepted`.
    """
    from src.inference.sp_artifact import (
        TENSOR_BUILD_ID_KEY,
        declares_binding,
        kg_binding_state,
        read_binding,
        read_sidecar,
        sidecar_path,
    )
    from src.inference.sp_index import SPArtifactError

    table = workspace / "shortest_paths.pt"
    if not table.exists() and not sidecar_path(table).exists():
        absent = {"verdict": "absent", "reason": "no table: served without shortest paths"}
        return {"source_binding": absent, "served": dict(absent), "tensor_checked": False}
    if kg_digest is None:
        why = "the workspace's graph is refused"
        return {"source_binding": _not_judged(why), "served": _not_judged(why),
                "tensor_checked": False}

    def check() -> None:
        if not table.exists():
            raise SPArtifactError("the sidecar is here and the table is not")
        sidecar = read_sidecar(table)
        meta = sidecar.meta if sidecar else None
        # Where the sidecar declares a binding, the tensor's build id is taken as
        # present; whether it is, and matches, is the service loader's check.
        keys = (TENSOR_BUILD_ID_KEY,) if declares_binding(meta, ()) else ()
        binding = read_binding(meta, keys, source=table.name)
        if kg_binding_state(binding, kg_digest, source=table.name) == "unrecorded":
            raise SPArtifactError(
                f"{table.name} carries no source binding, so the graph it was "
                "computed from is unrecorded (contract M2.5)"
            )

    binding = _judge(check, root)
    if binding["verdict"] == "refused":
        served = {"verdict": "refused", "reason": "its source binding is refused"}
    else:
        served = _not_judged(
            "the tensor is not loaded: its columns and values, its build id against "
            "the sidecar's and the hop bound are the service loader's checks"
        )
    return {"source_binding": binding, "served": served, "tensor_checked": False}


#: What a checkpoint's verdicts cover, stated in the report beside them.
CHECKPOINT_SCOPE = (
    "M2's source-contract checks only: whether the checkpoint's training record "
    "names this workspace's graph (serve_and_measure, M2.4) and this workspace's "
    "training inputs (resume, M2.3). Not whether its weights build a model over the "
    "graph, whether a device has the capacity, or whether the service runs."
)


def _checkpoint_verdicts(
    path: Path, root: Path, bound: Optional[Dict[str, str]], inputs: Any,
    workspace_refused: str = "the workspace's graph is refused",
) -> Dict[str, Any]:
    """One checkpoint, read once (`read_checkpoint`, the safe loader on CPU, as
    M1-M3 opens them), judged from that read; its digest is that read's."""
    from src.kg.artifacts import GRAPH_ARTIFACTS
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
        serve = _judge(lambda: verify_training_record(
            checkpoint, bound, tuple(GRAPH_ARTIFACTS),
            recorded_by=f"checkpoint {name}", compared_with="this workspace's graph",
        ), root)
    if inputs is None:
        resume = _not_judged("training refuses this workspace")
    else:
        from scripts.train_model import check_resume_parent

        resume = _judge(lambda: check_resume_parent(
            checkpoint, inputs, recorded_by=f"checkpoint {name}",
            compared_with="a run over this workspace",
        ), root)
    return {"sha256": read.identity.sha256,
            "m2_contract": {"serve_and_measure": serve, "resume": resume}}


def inventory_workspace(workspace: Path, root: Path) -> Dict[str, Any]:
    """One workspace: its graph, its dataset record, its training inputs, its
    table, its checkpoints."""
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
    entry["shortest_paths"] = _shortest_paths(
        workspace, bound["kg"] if bound else None, root
    )
    # Serving and measurement refuse the workspace when either check does
    # (contract M3b), so no checkpoint is judged servable over a refused record.
    served_bound, refused = bound, "the workspace's graph is refused"
    if bound is not None and entry["dataset_record"]["verdict"] != "accepted":
        served_bound, refused = None, "the workspace's dataset record is refused"
    checkpoints = sorted(
        path for path in workspace.rglob("*.pt")
        if not (path.parent == workspace and path.name in _NOT_CHECKPOINTS)
    )
    entry["checkpoints"] = {
        path.relative_to(root).as_posix(): _checkpoint_verdicts(
            path, root, served_bound, inputs, refused)
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
        for part in ("source_binding", "served"):
            counts[f"shortest_paths {part}:{entry['shortest_paths'][part]['verdict']}"] += 1
        for verdicts in entry["checkpoints"].values():
            for use, verdict in verdicts["m2_contract"].items():
                counts[f"checkpoint m2_contract {use}:{verdict['verdict']}"] += 1
    return {
        "fact": "M2.6 inventory",
        "what_this_shows": (
            "for each workspace, whether contract M2 accepts its graph, its training "
            "inputs, its shortest-path table's source binding and each checkpoint's "
            "training record, and whether contract M3b accepts its dataset record "
            "(binding state and origin apart), and why not"
        ),
        "read_only": (
            "nothing was moved, deleted or rewritten; refused artifacts are archived "
            "by the owner, not deleted"
        ),
        "checkpoint_verdicts_cover": CHECKPOINT_SCOPE,
        "not_checked": [
            "the shortest-path tensor (not loaded): whether a table with an accepted "
            "source binding would be served is not judged",
            "checkpoints the safe loader cannot open (listed as not judged)",
        ],
        "counts": dict(sorted(counts.items())),
        "workspaces": listing,
        "runtime": _runtime_facts(),
        "deployment_relationship": relationship,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1-M3 checkpoint family evidence")
    subject = parser.add_mutually_exclusive_group(required=True)
    subject.add_argument("--checkpoint-dir", type=Path,
                         help="One directory of *.pt checkpoints. Not searched recursively: "
                              "an architecture family is one directory, and mixing two "
                              "would make the counts describe no family in particular.")
    subject.add_argument("--inventory", type=Path,
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

    subject = args.inventory or args.checkpoint_dir
    if not subject.is_dir():
        raise SystemExit(f"not a directory: {subject}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.output} exists. Pass --overwrite to replace it, or write "
            "elsewhere — an evidence file that was silently replaced is not evidence."
        )

    if args.inventory:
        report = build_inventory(args.inventory, args.deployment_relationship)
    else:
        report = build_report(args.checkpoint_dir, args.deployment_relationship)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    if args.inventory:
        logger.info("%s -> %s", report["counts"], args.output)
        return 0
    summary = report["summary"]
    logger.info(
        "%d checkpoints, %d loaded, %d unreadable -> %s",
        summary["n_checkpoints_found"], summary["n_loaded"],
        summary["n_unreadable"], args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
