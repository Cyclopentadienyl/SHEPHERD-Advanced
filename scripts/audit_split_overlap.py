#!/usr/bin/env python
"""
M4 evidence — how much of the validation cohort's disease set is in training.
=============================================================================
Backlog item 10. M4 was measured once in a review thread: **100% of validation
diseases appear in training**, 7,970 of 7,970, over 100,000 training samples
across 10,576 diseases and 15,000 validation samples across 7,970.

That figure bounds every number this project reports on `val`, and item 2 puts it
into user-facing help — which is exactly why it must be reproducible from an
artifact rather than quoted from a conversation.

**Counts and hashes only.** BACKLOG §5.2 forbids patient ids, sample ids and
per-disease lists here, and nothing in the claim needs them: it is two set sizes
and the size of their intersection. The split digests are what let a later reader
confirm the numbers describe the files they have.

Usage:
    python scripts/audit_split_overlap.py \\
        --data-dir data/workspaces/<workspace> \\
        --output docs/working/EVIDENCE_M4_split_overlap.json

Module: scripts/audit_split_overlap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def disease_ids(data_dir: Path, split: str) -> List[int]:
    """Every sample's disease id in one split, in file order.

    Read through `src.kg.storage.file_storage.read_samples`, which is what the
    training and measurement paths read with — a second parser here could differ
    from them in exactly the way this evidence exists to rule out. It also refuses
    a missing split and names the ones that exist, so a workspace without the
    split asked for produces an error rather than an empty set silently reported
    as zero overlap.
    """
    from src.kg.storage.file_storage import read_samples

    return [int(sample.disease_id) for sample in read_samples(data_dir, split)]


def build_report(data_dir: Path, splits: List[str], platform_note: Optional[str]) -> Dict[str, Any]:
    from src.utils.fingerprint import file_sha256

    train_split, eval_split = splits
    train_ids = disease_ids(data_dir, train_split)
    eval_ids = disease_ids(data_dir, eval_split)
    train_set, eval_set = set(train_ids), set(eval_ids)
    shared = eval_set & train_set

    return {
        "fact": "M4",
        "what_this_shows": (
            f"how much of the {eval_split} cohort's disease set already appears in "
            f"{train_split}, which bounds what any {eval_split} metric can be evidence of"
        ),
        "splits": {"training": train_split, "evaluation": eval_split},
        "digests": {
            f"{split}_samples": file_sha256(data_dir / f"{split}_samples.json")
            for split in splits
        },
        "counts": {
            f"{train_split}_samples": len(train_ids),
            f"{train_split}_diseases": len(train_set),
            f"{eval_split}_samples": len(eval_ids),
            f"{eval_split}_diseases": len(eval_set),
            "shared_diseases": len(shared),
            f"{eval_split}_diseases_absent_from_{train_split}": len(eval_set - train_set),
        },
        # The ratio is stated as a fraction as well as a percentage, because a
        # percentage alone loses the denominator and the denominator is half the
        # claim: "100%" of 7,970 and "100%" of 12 are not the same finding.
        "overlap": {
            "shared_over_evaluation": (
                len(shared) / len(eval_set) if eval_set else None
            ),
            "as_written": (
                f"{len(shared)} of {len(eval_set)}" if eval_set else "no evaluation diseases"
            ),
        },
        "platform_note": platform_note,
        "excluded_by_design": [
            "patient ids",
            "sample ids",
            "per-disease lists (only set sizes are recorded)",
        ],
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M4 split disease-overlap evidence")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs=2, default=["train", "val"],
                        metavar=("TRAINING", "EVALUATION"),
                        help="Which two splits to compare, training first. Defaults to "
                             "train and val, which is the pair the current trainer "
                             "selects checkpoints on. A supplied test cohort goes here "
                             "as the second argument.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing --output. Off by default.")
    parser.add_argument("--platform-note", default=None,
                        help="An operator's statement about this machine's relationship "
                             "to the deployment. Recorded verbatim and not verified.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    report = build_report(args.data_dir, args.splits, args.platform_note)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    logger.info("%s -> %s", report["overlap"]["as_written"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
