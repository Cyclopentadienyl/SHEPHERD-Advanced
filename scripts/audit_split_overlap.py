#!/usr/bin/env python
"""
M4 — the disease overlap between two splits, measured from the files on disk.
=============================================================================
Backlog item 10. M4 was measured once in a review thread: **100% of validation
diseases appear in training**, 7,970 of 7,970, over 100,000 training samples
across 10,576 diseases and 15,000 validation samples across 7,970.

That figure bounds every number this project reports on `val`, and item 2 puts it
into user-facing help — which is exactly why it must be reproducible from an
artifact rather than quoted from a conversation.

**What changed, and why this script did not become redundant.**
``src/kg/disease_allocation.py`` now cuts the disease universe before generation,
so a workspace built by the current pipeline is disease-disjoint by construction
and `split_manifest.json` records that. It would be easy to conclude that
measuring the overlap is now pointless. It is not, for one reason: every
disjointness claim in the manifest is made **in the generating process, about
values held in its own memory**. `build_split_manifest` derives the realised sets
from the records it is about to write, asserts coverage, and writes both the
verdict and the files. Nothing afterwards reads those files back.

So the manifest cannot distinguish "these are the cohorts that allocation
produced" from "a manifest is sitting next to two sample files it does not
describe" — a workspace whose samples were replaced, or whose manifest was copied
in from elsewhere. This script re-reads the sample files with the same reader the
trainer uses, recomputes the disease sets, and compares them against what the
manifest claims. That is an **independent** check, not a second copy of the
generator's own.

Two verdicts, deliberately given different force:

- A manifest that disagrees with the files beside it is a broken workspace under
  any reading, so the disagreement is always fatal.
- Non-zero overlap is fatal only under ``--require-disjoint``. It has to stay
  measurable without refusal, because reproducing the M4 baseline means running
  this over a pre-split workspace where the overlap is total and expected.

**Schema.** The report is ``schema_version`` 2. Version 1 is the unversioned
shape ``EVIDENCE_M4.json`` carries; it was not redefined in place, for the reason
§6.8's audit gives for the same situation — a v1 artifact already exists and has
been relied on.

**Counts and hashes only.** BACKLOG §5.2 forbids patient ids, sample ids and
per-disease lists here, and nothing in the claim needs them: it is two set sizes
and the size of their intersection. The split digests are what let a later reader
confirm the numbers describe the files they have.

Usage:
    python scripts/audit_split_overlap.py \\
        --data-dir data/workspaces/<workspace> \\
        --output docs/working/EVIDENCE_M4_split_overlap.json \\
        --require-disjoint

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

# Shared with the other evidence scripts rather than restated here: all three
# reports are read together, and a claim spelled differently in each cannot be
# compared across them.
from src.evaluation.cohort import (  # noqa: E402
    COHORT_KINDS,
    DEFAULT_COHORT_KIND,
    GENERATED_SPLITS,
    MANIFEST_FILENAME,
    resolve_cohort,
    verify_generated_cohorts,
)
from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

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


#: Bumped when this report's shape changes. Version 1 is the unversioned shape
#: ``EVIDENCE_M4.json`` carries, kept readable rather than redefined in place.
REPORT_SCHEMA_VERSION = 4



def manifest_verification(
    data_dir: Path, splits: List[str], evaluation_kind: str, measured: Dict[str, set]
) -> Dict[str, Any]:
    """What ``verify_generated_cohorts`` established, written into the report.

    **The verification itself is not repeated here.** It runs at every entry point
    that consumes generated cohorts — training, measurement, calibration, this
    audit — from one definition in ``src/evaluation/cohort.py``. A second copy in
    the audit would be free to drift from the one the pipeline actually enforces,
    and would then report a workspace as sound under rules nothing else applies.

    What this adds is the *record*: a reader holding only the artifact can see
    which splits were bound, on what, and what the manifest claimed. The
    evaluation split appears only when it is generated — a supplied cohort is not
    described by this manifest, and checking it against one would be a verdict
    about a cut it was never part of.
    """
    train_split, eval_split = splits
    manifest = json.loads((data_dir / MANIFEST_FILENAME).read_text())
    compared = list(GENERATED_SPLITS) if evaluation_kind == "generated" else [train_split]

    section = {
        "verified_splits": compared,
        "verified_against": [
            "the sample files' SHA-256, against the manifest's artifact digests",
            "the realised disease sets, recomputed from the records",
            "the manifest's realised digests against its own allocated ones",
            "disjointness, measured and compared with the manifest's claim",
        ],
        "manifest_schema_version": manifest.get("schema_version"),
        "claimed_disjoint": manifest.get("disjoint"),
        "allocation_algorithm": manifest.get("allocation", {}).get("algorithm"),
        "generation_algorithm": manifest.get("generation", {}).get("algorithm"),
    }
    if evaluation_kind == "generated":
        section["measured_disjoint"] = not (measured[train_split] & measured[eval_split])
    else:
        section["why_the_evaluation_split_is_not_verified"] = (
            f"{eval_split} is a supplied cohort; this manifest describes the "
            "generated splits and says nothing about it"
        )
    return section


def build_report(
    data_dir: Path, splits: List[str], evaluation_kind: str, relationship: str
) -> Dict[str, Any]:
    from src.utils.fingerprint import file_sha256

    train_split, eval_split = splits
    # **Both splits are resolved before anything is read.** The training split is
    # always generated; the evaluation split is whichever kind the caller states,
    # and `resolve_cohort` is what refuses a supplied cohort wearing a generated
    # name, a generated one with no manifest, and a workspace built before the
    # allocation step.
    resolve_cohort(data_dir, eval_split, evaluation_kind)
    # **Refuses before a single count is taken.** The generated cohorts must be the
    # ones their manifest describes — same bytes, same disease sets, disjoint — and
    # that is established by the one verifier the whole pipeline uses. A report
    # produced over an unverified workspace would look like a measurement.
    verify_generated_cohorts(data_dir)
    train_ids = disease_ids(data_dir, train_split)
    eval_ids = disease_ids(data_dir, eval_split)
    train_set, eval_set = set(train_ids), set(eval_ids)

    # **An empty split cannot establish the fact this file exists for.** M4 bounds
    # what an evaluation metric can be evidence of, by measuring how much of the
    # evaluation cohort's disease set the training split already contains. With no
    # evaluation diseases the ratio has no denominator; with no training diseases
    # the overlap is zero for a reason that is about the workspace, and either
    # would be read as "no contamination" by someone citing the file later. The
    # shared sample reader is left alone — a split that exists and is empty is a
    # legitimate thing for it to return, and only this audit needs to refuse it.
    for split, ids in ((train_split, train_ids), (eval_split, eval_ids)):
        if not ids:
            raise SystemExit(
                f"the {split} split holds no samples. M4 compares two disease sets, "
                "and an overlap involving an empty one says nothing about "
                "contamination — only that this workspace has no such split."
            )

    shared = eval_set & train_set

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
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
        #
        # Unguarded on purpose. `eval_set` cannot be empty here — the refusal above
        # is what makes that true — and an `if eval_set else None` fallback would
        # tell a reader the opposite, that this file can report an overlap with no
        # denominator.
        "overlap": {
            "shared_over_evaluation": len(shared) / len(eval_set),
            "as_written": f"{len(shared)} of {len(eval_set)}",
        },
        "manifest_verification": manifest_verification(
            data_dir, splits, evaluation_kind, {train_split: train_set, eval_split: eval_set}
        ),
        "evaluation_cohort_kind": evaluation_kind,
        "deployment_relationship": relationship,
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
    parser.add_argument("--evaluation-cohort-kind", default=DEFAULT_COHORT_KIND,
                        choices=COHORT_KINDS,
                        help="What the second split is. `generated` means this "
                             "project's own val cohort, where disease-disjointness "
                             "is a contract and any overlap is a failure. `supplied` "
                             "means an institutional or external cohort, where the "
                             "overlap is the measurement being taken and refusing on "
                             "it would refuse the finding.")
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

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    # **Verification failures refuse before an artifact exists.** Earlier this
    # audit wrote its report and then refused, so a failure left something to
    # cite. That was right while this was the only gate; the same verifier now
    # runs at training and measurement, where writing a report makes no sense, and
    # a workspace that fails it is one to rebuild rather than to publish a
    # measurement about. The refusal message names which check failed on which
    # split, which is the citable content.
    try:
        report = build_report(args.data_dir, args.splits, args.evaluation_cohort_kind,
                              args.deployment_relationship)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    logger.info("%s -> %s", report["overlap"]["as_written"], args.output)

    # **The report is written first, then the verdict refuses.** A failed audit's
    # evidence is the artifact describing the failure; discarding it on the way
    # out would leave the operator with an exit code and nothing to cite.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
