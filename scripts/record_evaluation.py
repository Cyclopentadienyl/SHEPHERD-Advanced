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

**Only a report the registry's report store holds** (D5, since M5): `--report`
names `<registry>/versions/<label>/reports/<run id>/report.json`, the path the
measurement printed as `Report:`. A copy elsewhere is refused, saying to
register the store's original: a record names the run directory its report
was kept in, and a copy is no run. **Registered by the verified run** (policy
rev 16, K8-6r.2): only this report is read, and every verified run gets a
record of its own; the same run's same report again writes nothing, and the
same run with other bytes is refused (a run's report is sealed when it is
published). How its record differs from the other records of its condition
group, and a possible overlap with a record merged before each run had its own
record, are printed as notes that decide nothing. The registry is found from the
report's own path and **must be the deployment's**: the one `SHEPHERD_REGISTRY`
names, compared by resolved path, so set the variable in this shell as the
service's environment sets it (deployment guide §4.11). With the variable unset,
and for a report in another directory (a copy of the registry, say), the
registration is refused before anything in that registry is read. A registry
moved as a whole still works once `SHEPHERD_REGISTRY` names its new place.

**The ledger has to be beside the weights.** Name the checkpoint with exactly one
of `--checkpoint FILE`, which must hold the bytes the report measured (any file
name), or `--checkpoint-dir DIR`, which must hold such a `*.pt` file; otherwise
nothing is written. **The ledger is `evaluations.json` in the directory of that
file's resolved path**, as for `measure_scorer.py --record` and the read-only
view: a checkpoint reached through a link files its ledger beside the file the
link names, which is where the service looks.

Usage:
    python scripts/record_evaluation.py \\
        --report <registry>/versions/<label>/reports/<run>/report.json \\
        --checkpoint <workspace>/runs/<arch>/<run>/checkpoints/best.pt

    python scripts/record_evaluation.py \\
        --report <registry>/versions/<label>/reports/<run>/report.json \\
        --checkpoint-dir <workspace>/runs/<arch>/<run>/checkpoints

    # what does a ledger hold for one set of weights? With --checkpoint, the
    # ledger the service reads for that file; with --checkpoint-dir, the one in
    # that directory.
    python scripts/record_evaluation.py \\
        --checkpoint <workspace>/runs/<arch>/<run>/checkpoints/best.pt \\
        --show <checkpoint sha256>

`--show` prints the records as the ledger holds them (JSON on stdout), and
for each one its run, "not recorded" for a record merged before each run had
its own record, and its possible overlaps with records naming a run (on the
log); a record whose run identity cannot be read is said with its reason.

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
from src.evaluation.registration import (  # noqa: E402
    RegistrationRefused,
    group_note,
    historical_note,
    register_report,
)
from src.evaluation.sidecar import (  # noqa: E402
    LEDGER_FILENAME,
    possible_overlaps,
    read_ledger,
    record_identities,
    records_for,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record or read evaluation results in the ledger beside a checkpoint"
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--checkpoint", type=Path, default=None, metavar="FILE",
                       help=f"The checkpoint file the report measured, under any name. "
                            f"It must hold the report's checkpoint bytes. The ledger is "
                            f"{LEDGER_FILENAME} in the directory of its resolved path, "
                            f"where the service reads it.")
    where.add_argument("--checkpoint-dir", type=Path, default=None, metavar="DIR",
                       help=f"A directory holding the checkpoint as a *.pt file, under "
                            f"the run layout <workspace>/runs/<arch>/<run>/checkpoints. "
                            f"With --report, the ledger is {LEDGER_FILENAME} in the "
                            f"directory of the matching file's resolved path; with "
                            f"--show, the {LEDGER_FILENAME} inside DIR is read.")
    parser.add_argument("--report", type=Path, default=None,
                        help="A measurement report of a registered cohort version in the "
                             "registry's report store, "
                             "<registry>/versions/<label>/reports/<run id>/report.json, as "
                             "scripts/measure_scorer.py keeps it, in the deployment's "
                             "registry (SHEPHERD_REGISTRY, which must be set here); a copy "
                             "elsewhere is refused. One record is derived from its manifest.")
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

    if args.show is not None:
        # The same single read a registration makes; nothing is written.
        if args.checkpoint is not None:
            ledger_path = args.checkpoint.resolve().parent / LEDGER_FILENAME
        else:
            ledger_path = args.checkpoint_dir.resolve() / LEDGER_FILENAME
        try:
            ledger, _ = read_ledger(ledger_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        found = records_for(ledger, args.show)
        print(json.dumps(found, indent=2, sort_keys=True))
        if not found:
            logger.info("%s: %s", ledger_path, RECORD_ABSENCE)
        for line in _run_lines(ledger, found):
            logger.info("%s", line)
        return 0

    try:
        result = register_report(
            args.report, checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir,
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
    else:
        logger.info("Registered record %d (%s) -> %s", sequence, where, result.ledger_path)
    # Two notes that decide nothing (policy rev 16): every run is kept.
    for note in (group_note(result), historical_note(result)):
        if note is not None:
            logger.info("%s", note)
    if result.usage_history_problem is not None:
        logger.warning("NOTE: %s.", result.usage_history_problem)
    return 0


def _run_lines(ledger: dict, shown: list) -> list:
    """One line per shown record: its sequence and its run, as the file's
    run-identity boundary reads it (`record_identities`): "not recorded" for a
    record merged before each run had its own record, or why its identity
    cannot be read; then its possible overlaps (`possible_overlaps`), never
    confirmed. The two kinds of record stay apart."""
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    by_record = {id(record): (identity, overlaps) for record, identity, overlaps in
                 zip(records, record_identities(ledger), possible_overlaps(ledger))}
    lines = []
    for record in shown:
        identity, overlaps = by_record[id(record)]
        registration = record.get("registration")
        sequence = registration.get("sequence") if isinstance(registration, dict) else None
        name = f"record {sequence}" if type(sequence) is int else "a record without a sequence"
        if identity.problem is not None:
            lines.append(f"{name}: its run cannot be read: {identity.problem}")
            continue
        run = identity.run_id or ("not recorded (registered before each run had its own "
                                  "record)")
        line = f"{name}: run {run}"
        if overlaps:
            line += (f"; possible overlap with record{'s' if len(overlaps) != 1 else ''} "
                     f"{', '.join(str(n) for n in overlaps)} (not recorded)")
        lines.append(line)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
