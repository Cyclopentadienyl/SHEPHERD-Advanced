#!/usr/bin/env python
"""
Put a measurement's result into the ledger beside the checkpoint it describes.
==============================================================================
`EVALUATION_COHORTS.md` §6.5; `PLAN_TEST_RESULTS.md` D5 and §6 phase 1, step 2.

**Registration later, by hand, from a kept report.** `measure_scorer.py --record`
registers a report right after its run; this entry point registers one that was
kept without being registered — because nobody asked, or because the checkpoint
directory is the institution's and was read-only to whoever measured. Both call
`src/evaluation/registration.register_report`, so a report registered either way
is the same record.

The record is derived from the report, never from the command line: a run's own
manifest is what says which checkpoint, which cohort version and which mode it
was, and re-asserting any of that here would let the ledger describe a run that
did not happen. Only a registered cohort version's report is registered; a
generated split's is refused.

**The ledger has to be beside the weights.** `--checkpoint-dir` must hold a file
whose bytes are the checkpoint the report measured; otherwise nothing is written.

Usage:
    python scripts/record_evaluation.py \\
        --report <registry>/versions/<label>/reports/<run>/report.json \\
        --checkpoint-dir <workspace>/runs/<arch>/<run>/checkpoints

    # what does this ledger hold for one set of weights?
    python scripts/record_evaluation.py \\
        --checkpoint-dir <workspace>/runs/<arch>/<run>/checkpoints \\
        --show <checkpoint sha256>

Every refusal exits non-zero with one message that says nothing was written.

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

from src.evaluation.caveats import RECORD_ABSENCE  # noqa: E402
from src.evaluation.registration import RegistrationRefused, register_report  # noqa: E402
from src.evaluation.sidecar import LEDGER_FILENAME, read_ledger, records_for  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or read evaluation results beside a checkpoint directory"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help=f"Directory holding the checkpoint, under the run layout "
                             f"<workspace>/runs/<arch>/<run>/checkpoints. The ledger "
                             f"is {LEDGER_FILENAME} inside it.")
    parser.add_argument("--report", type=Path, default=None,
                        help="A measurement report of a registered cohort version, as "
                             "scripts/measure_scorer.py keeps it in the registry's "
                             "report store. One record is derived from its manifest.")
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

    checkpoint_dir = args.checkpoint_dir.resolve()

    if args.show is not None:
        # The same single read a registration makes; nothing is written.
        ledger_path = checkpoint_dir / LEDGER_FILENAME
        try:
            ledger, _ = read_ledger(ledger_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        found = records_for(ledger, args.show)
        print(json.dumps(found, indent=2, sort_keys=True))
        if not found:
            logger.info("%s: %s", ledger_path, RECORD_ABSENCE)
        return 0

    try:
        result = register_report(
            args.report, ledger_dir=checkpoint_dir, verify_checkpoint_beside=True,
        )
    except RegistrationRefused as exc:
        raise SystemExit(str(exc)) from exc

    record = result.record
    sequence = record["registration"]["sequence"]
    where = (f"{record['cohort_role']} · {record['cohort']['label']} for checkpoint "
             f"{record['checkpoint_digest'][:12]}...")
    if result.outcome == "already_registered":
        logger.info("%s already holds this report as record %d (%s); nothing written.",
                    result.ledger_path, sequence, where)
    elif result.outcome == "corroborated":
        logger.info("This report agrees with record %d (%s) in %s; its digest was added "
                    "to that record's corroborating reports.",
                    sequence, where, result.ledger_path)
    else:
        logger.info("Registered record %d (%s) -> %s", sequence, where, result.ledger_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
