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


def _filename_number(name: str) -> Optional[float]:
    """The score baked into a checkpoint filename, or `None`.

    `model-45-0.6975.pt` carries `0.6975`. Parsed positionally — the last
    dot-decimal token before the extension — rather than by a naming convention
    this script would then be asserting rather than observing.
    """
    stem = name[: -len(".pt")] if name.endswith(".pt") else name
    for token in reversed(stem.split("-")):
        try:
            value = float(token)
        except ValueError:
            continue
        return value
    return None


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    """One checkpoint, reduced to the aggregate facts M1-M3 are about."""
    import torch

    from src.utils.checkpoint_paths import ranking_score_detail

    record: Dict[str, Any] = {
        "filename": path.name,
        "filename_number": _filename_number(path.name),
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
        record["load_error"] = f"{type(exc).__name__}: {exc}"
        return record

    if not isinstance(checkpoint, dict):
        record["load_error"] = f"not a dict: {type(checkpoint).__name__}"
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

    Counters rather than per-checkpoint lists wherever a count answers the
    question, so the file stays small and stays publishable.
    """
    loaded = [r for r in records if r["loaded"]]

    key_presence = Counter()
    for record in loaded:
        key_presence.update(record["keys_present"])

    widths = Counter()
    for record in loaded:
        widths.update(record["in_channels"].values())

    metrics = Counter(r["logs_metric"] for r in loaded if r["logs_metric"])

    # M3: does the filename's number match the logs' ranking metric? Compared at
    # the precision a filename carries, since `model-45-0.6975.pt` is a rounded
    # rendering of `0.69754...` and an exact comparison would report a mismatch
    # that is only a formatting difference.
    agreements, disagreements, uncomparable = 0, 0, 0
    for record in loaded:
        number, value = record["filename_number"], record["logs_value"]
        if number is None or value is None:
            uncomparable += 1
        elif abs(number - value) < 10 ** -_decimals(number):
            agreements += 1
        else:
            disagreements += 1

    return {
        "n_checkpoints_found": len(records),
        "n_loaded": len(loaded),
        "n_unreadable": len(records) - len(loaded),
        "key_presence_counts": dict(sorted(key_presence.items())),
        "in_channels_value_counts": {str(k): v for k, v in sorted(widths.items())},
        "logs_ranking_metric_counts": dict(sorted(metrics.items())),
        "filename_vs_logs": {
            "agree": agreements,
            "disagree": disagreements,
            "uncomparable": uncomparable,
        },
    }


def _decimals(value: float) -> int:
    """Decimal places a filename number was written to, so a rounded rendering is
    compared as one."""
    text = repr(value)
    return len(text.split(".")[1]) if "." in text else 0


def _runtime_facts() -> Dict[str, Any]:
    """What this ran on, without saying **where** or **who**.

    §5.2 forbids operator and host names, and that is not in tension with
    traceability: what an institutional reader needs is that the evidence came
    from a machine of the deployment's kind, which is a statement about hardware
    and software rather than about a hostname. The narrow checkable facts are
    recorded here; the claim that this machine *is* equivalent to the deployment
    is `--platform-note`, made by a person who can make it. The same split
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


def build_report(checkpoint_dir: Path, platform_note: Optional[str]) -> Dict[str, Any]:
    from src.utils.fingerprint import file_sha256

    paths = sorted(checkpoint_dir.glob("*.pt"))
    records = [inspect_checkpoint(path) for path in paths]

    return {
        "fact": "M1-M3",
        "what_this_shows": (
            "which keys a family of checkpoints carries, the input width they were "
            "built with, and whether the number in a filename is the ranking metric "
            "in its logs"
        ),
        "checkpoint_digests": {path.name: file_sha256(path) for path in paths},
        "summary": summarise(records),
        "per_checkpoint": [
            {
                "filename": r["filename"],
                "loaded": r["loaded"],
                "load_error": r["load_error"],
                "keys_present": r["keys_present"],
                "in_channels": r["in_channels"],
                "logs_metric": r["logs_metric"],
                "logs_value": r["logs_value"],
                "filename_number": r["filename_number"],
            }
            for r in records
        ],
        "runtime": _runtime_facts(),
        "platform_note": platform_note,
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
    parser.add_argument("--platform-note", default=None,
                        help="An operator's statement about this machine's relationship "
                             "to the deployment — for example that it is an identical "
                             "sibling. Recorded verbatim. This script cannot verify it "
                             "and does not try to.")
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

    report = build_report(args.checkpoint_dir, args.platform_note)
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
