#!/usr/bin/env python
"""
Put a measurement's result into the ledger beside the checkpoints it describes.
===============================================================================
`EVALUATION_COHORTS.md` §6.5, step 5 of §6.6.

**A separate entry point rather than a step inside `measure_scorer.py`,** for two
reasons. The checkpoint directory is the institution's and may be read-only to
whoever runs a measurement, so writing into it has to be a decision someone makes
rather than a side effect of measuring. And the ledger can then be built from
artifacts that already exist, including ones produced before it did, without
re-running anything.

The record is derived from the measurement artifact, never from the command line:
a run's own manifest is what says which checkpoint, which cohort and which mode
it was, and re-asserting any of that here would let the ledger describe a run
that did not happen.

Usage:
    python scripts/record_evaluation.py \\
        --report artifacts/mode_a.json \\
        --checkpoint-dir data/workspaces/<ws>/checkpoints

    # what does this ledger hold for one set of weights?
    python scripts/record_evaluation.py \\
        --checkpoint-dir data/workspaces/<ws>/checkpoints \\
        --show <checkpoint sha256>

Module: scripts/record_evaluation.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sidecar import (  # noqa: E402
    LEDGER_FILENAME,
    append_record,
    build_record,
    find_checkpoint,
    ledger_digest,
    read_ledger,
    records_for,
    write_ledger,
)
from src.utils.fingerprint import file_sha256  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or read evaluation results beside a checkpoint directory"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help=f"Directory holding the checkpoints. The ledger is "
                             f"{LEDGER_FILENAME} inside it.")
    parser.add_argument("--report", type=Path, default=None,
                        help="A measurement artifact as scripts/measure_scorer.py "
                             "writes it. One record is derived from its manifest.")
    parser.add_argument("--show", default=None, metavar="CHECKPOINT_DIGEST",
                        help="Print what the ledger holds for one checkpoint digest. "
                             "An empty result means this ledger holds no record for "
                             "those weights — not that they were never evaluated.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if (args.report is None) == (args.show is None):
        raise SystemExit("pass exactly one of --report and --show")

    ledger_path = args.checkpoint_dir / LEDGER_FILENAME
    ledger = read_ledger(ledger_path)
    digest_at_read = ledger_digest(ledger_path)

    if args.show is not None:
        found = records_for(ledger, args.show)
        print(json.dumps(found, indent=2, sort_keys=True))
        if not found:
            logger.info(
                "%s holds no record for %s. That is a statement about this ledger, "
                "not about whether these weights were ever evaluated.",
                ledger_path, args.show,
            )
        return 0

    if not args.report.exists():
        raise SystemExit(f"{args.report} does not exist")
    report = json.loads(args.report.read_text())
    try:
        record = build_record(report, file_sha256(args.report))
    except KeyError as exc:
        raise SystemExit(
            f"{args.report} is missing {exc}, so it is not a measurement artifact "
            "this can derive a record from"
        ) from exc

    # **The ledger has to be beside the weights it describes.** Without this it
    # can be written into any directory, including one holding no checkpoint at
    # all, and a reader looking up a digest in the directory they were handed
    # would find a record filed under the wrong address.
    checkpoint = find_checkpoint(args.checkpoint_dir, record["checkpoint_digest"])
    if checkpoint is None:
        raise SystemExit(
            f"{args.checkpoint_dir} holds no checkpoint whose bytes are "
            f"{(record['checkpoint_digest'] or 'unknown')[:12]}..., which is what "
            f"{args.report} measured. Point --checkpoint-dir at the directory "
            "holding those weights; a ledger beside a different checkpoint is a "
            "record filed under the wrong address."
        )

    try:
        updated = append_record(ledger, record)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if updated is ledger:
        logger.info("%s already holds this exact record; nothing written.", ledger_path)
        return 0

    try:
        write_ledger(ledger_path, updated, expected_digest=digest_at_read)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info(
        "Recorded mode %s on cohort %s for %s -> %s",
        record["mode"], record["cohort_role"], checkpoint.name, ledger_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
