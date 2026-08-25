#!/usr/bin/env python
"""
M1-M3 evidence — what a family of checkpoints actually carries.
===============================================================
Backlog item 10. Three facts were established by reading checkpoints in a review
thread and have existed only as pasted text since:

  M1  no scanned checkpoint carries `metadata` or `in_channels_dict`
  M2  the input width is 128, where the frozen evaluator's hardcoded fallback
      is 256 — the size mismatch it dies on
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
the frozen evaluator's loader fails on this family, so a scan that stopped at the
first unreadable file would destroy the evidence it was run to collect.

Usage:
    python scripts/audit_checkpoint_family.py \\
        --checkpoint-dir checkpoints/hgt \\
        --output docs/working/EVIDENCE_M1_M3_checkpoints.json

Module: scripts/audit_checkpoint_family.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The vocabulary is shared with the other evidence scripts rather than restated
# here: an institutional reader joins these reports by machine, and that join
# breaks the moment two scripts spell the same claim differently.
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
    """What this ran on, without saying **where** or **who**.

    §5.2 forbids operator and host names, and that is not in tension with
    traceability: what an institutional reader needs is that the evidence came
    from a machine of the deployment's kind, which is a statement about hardware
    and software rather than about a hostname. The narrow checkable facts are
    recorded here; the claim that this machine *is* equivalent to the deployment
    is `--deployment-relationship`, chosen from a bounded vocabulary by a person
    who can make it. The same split
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1-M3 checkpoint family evidence")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="One directory of *.pt checkpoints. Not searched recursively: "
                             "an architecture family is one directory, and mixing two "
                             "would make the counts describe no family in particular.")
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

    if not args.checkpoint_dir.is_dir():
        raise SystemExit(f"not a directory: {args.checkpoint_dir}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"{args.output} exists. Pass --overwrite to replace it, or write "
            "elsewhere — an evidence file that was silently replaced is not evidence."
        )

    report = build_report(args.checkpoint_dir, args.deployment_relationship)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

    summary = report["summary"]
    logger.info(
        "%d checkpoints, %d loaded, %d unreadable -> %s",
        summary["n_checkpoints_found"], summary["n_loaded"],
        summary["n_unreadable"], args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
