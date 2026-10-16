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
  any reading, so the disagreement is always fatal. So is overlap between the
  generated `train` and `val`: disjointness is the allocation's contract.
- Overlap with a registered test cohort is never fatal. That cohort was not cut
  from this disease universe, so the overlap is the measurement being taken, and
  refusing on it would refuse the finding.

**A test cohort is a registered cohort version** (contract M4). It is imported
and mapped to one graph by ``scripts/import_cohort.py`` and audited here with
``--registry DIR --cohort LABEL``: the version is resolved from one read of its
``cohort.json``, its ``samples.json`` is read once and refused unless its digest
is the one that manifest binds, and the version's recorded graph and dataset
record digests must be the ones this workspace's manifest records, because its
disease indices name nodes of the graph it was mapped against. The loose
``<split>_samples.json`` this mode used to read from the workspace is gone with
``--evaluation-cohort-kind``.

**Schema.** The report carries ``schema_version`` (`REPORT_SCHEMA_VERSION`).
Version 1 is the unversioned shape ``EVIDENCE_M4.json`` carries; it was not
redefined in place, for the reason §6.8's audit gives for the same situation — a
v1 artifact already exists and has been relied on.

**Counts and hashes only.** BACKLOG §5.2 forbids patient ids, sample ids and
per-disease lists here, and nothing in the claim needs them: it is two set sizes
and the size of their intersection. The split digests are what let a later reader
confirm the numbers describe the files they have. A registered version is named
by its label, role and digests, never by a path or a case.

Usage:
    python scripts/audit_split_overlap.py \\
        --data-dir data/workspaces/<workspace> \\
        --output docs/working/EVIDENCE_M4_split_overlap.json

    python scripts/audit_split_overlap.py \\
        --data-dir data/workspaces/<workspace> \\
        --registry <absolute registry directory> --cohort <label> \\
        --output <report>.json

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
    GENERATED_SPLITS,
    resolve_cohort,
    verify_cohort_reads,
)
from src.utils.file_modes import publish_bytes  # noqa: E402
from src.utils.provenance import DEPLOYMENT_RELATIONSHIPS, UNSTATED_RELATIONSHIP  # noqa: E402

logger = logging.getLogger(__name__)


def disease_ids(read: Any) -> List[int]:
    """Every sample's disease id in one split, in file order, from its one read.

    `read` is what `src.kg.storage.file_storage.read_samples` (or, for a
    registered version, `read_samples_file`) returned — the shared reader
    training and measurement use (contract M2.1) — so a second parser here
    cannot differ from them in exactly the way this evidence exists to rule out,
    and the ids counted come from the bytes whose digest the report records and
    the verifier compared.
    """
    return [int(sample.disease_id) for sample in read.samples]


#: Bumped when this report's shape changes. Version 1 is the unversioned shape
#: ``EVIDENCE_M4.json`` carries, kept readable rather than redefined in place.
#: v6 — the supplied cohort is a registered cohort version, recorded under
#: ``cohort_version`` in place of ``evaluation_cohort_kind: supplied``.
REPORT_SCHEMA_VERSION = 6

#: The two splits a generated audit compares when ``--splits`` is not given:
#: the pair the current trainer selects checkpoints on.
DEFAULT_SPLITS = ("train", "val")

#: The generated split a registered cohort version is compared with. Not a
#: choice: M4 asks how much of a test cohort's disease set the model had
#: labelled training examples for, and `train` is what training learns from
#: (`val` selects the checkpoint; the measurement report's R10 covers it).
TRAINING_SPLIT = "train"


def manifest_verification(
    verified: Any, splits: List[str], measured: Dict[str, set],
    cohort_label: Optional[str] = None,
) -> Dict[str, Any]:
    """What ``verify_cohort_reads`` established, written into the report.

    **The verification itself is not repeated here.** It runs at every entry point
    that consumes generated cohorts — training, measurement, this audit — from
    one definition in ``src/evaluation/cohort.py``. A second copy in
    the audit would be free to drift from the one the pipeline actually enforces,
    and would then report a workspace as sound under rules nothing else applies.

    What this adds is the *record*: a reader holding only the artifact can see
    which splits were bound, on what, and what the manifest claimed. The
    evaluation split appears only when it is generated — a registered cohort
    version (`cohort_label`) is not described by this manifest, and checking it
    against one would be a verdict about a cut it was never part of.
    """
    train_split, eval_split = splits
    # **From the verifier's own return value, not recomputed here.** A section
    # naming a scope this function derived independently could disagree with the
    # gate that ran, and the artifact would then misdescribe itself — which is the
    # defect this parameter exists to remove.
    manifest = verified.manifest
    # **The list describes the gate that ran, not the gate in general.** It used
    # to name disjointness unconditionally while the report also said
    # `disjointness_was_measured: false` three lines below — an artifact
    # contradicting itself about its own verification.
    checks = [
        "the sample files' SHA-256, against the manifest's artifact digests",
        "the realised disease sets, recomputed from the records",
        "the manifest's realised digests against its own allocated ones",
    ]
    if verified.disjointness_measured:
        checks.append("disjointness, measured between the two generated cohorts")
    if verified.disjointness_claim_checked:
        checks.append("the manifest's own disjointness claim")
    if len(verified.verified) == 1:
        checks = [
            line.replace("the sample files'", f"the {verified.verified[0]} file's")
            .replace("the realised disease sets, recomputed",
                     f"the realised {verified.verified[0]} disease set, recomputed")
            .replace("realised digests against its own allocated ones",
                     f"realised {verified.verified[0]} digest against its own allocated one")
            for line in checks
        ]

    section = {
        "verified_splits": list(verified.verified),
        "verified_against": checks,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_disjointness_claim": manifest.get("disjoint"),
        "allocation_algorithm": manifest.get("allocation", {}).get("algorithm"),
        "generation_algorithm": manifest.get("generation", {}).get("algorithm"),
    }
    section["disjointness_was_measured"] = verified.disjointness_measured
    section["manifest_disjointness_claim_checked"] = verified.disjointness_claim_checked
    if verified.disjointness_measured:
        section["measured_disjoint"] = not (measured[train_split] & measured[eval_split])
    else:
        section["why_the_evaluation_split_is_not_verified"] = (
            f"the evaluation cohort is the registered cohort version {cohort_label!r} "
            f"(role {eval_split}), not a split of this workspace, so this manifest "
            "says nothing about it; the version binds its own samples and was checked "
            "against this workspace's graph instead (see cohort_version). Generated "
            "val was not read, and the manifest's `disjoint` field describes the "
            "generated train/val relationship rather than this overlap, so neither "
            "the state of val nor that claim blocks or supports this report"
        )
    return section


def build_report(data_dir: Path, splits: List[str], relationship: str) -> Dict[str, Any]:
    """Two of the generator's own splits: both read, both verified, compared."""
    from src.kg.artifacts import read_split_manifest
    from src.kg.storage.file_storage import read_samples

    # **Both splits are resolved before anything is read.** `resolve_cohort` is
    # what refuses a name that is not an identifier, a name that is not one of
    # the generator's splits (a test cohort is a registered version, audited
    # with --registry and --cohort), a generated split with no manifest, and a
    # workspace built before the allocation step. The training split goes
    # through it too: a name only ever reaches the reader once it is resolved.
    for split in dict.fromkeys((*splits, *GENERATED_SPLITS)):
        resolve_cohort(data_dir, split)
    # **Refuses before a single count is taken, over exactly what it reads.** The
    # generated cohorts must be the ones their manifest describes — same bytes,
    # same disease sets — and that is established by the one verifier the whole
    # pipeline uses.
    #
    # **One read per file (contract M2.1).** The manifest is read once and every
    # split this audit uses once; the verifier compares those reads, the counts
    # are taken from them, and the digests recorded are theirs. Nothing is hashed
    # by path, so a file replaced in between is either counted and recorded as
    # read or refused as not the file the manifest describes.
    manifest = read_split_manifest(data_dir)
    reads = {split: read_samples(data_dir, split) for split in GENERATED_SPLITS}
    verified = verify_cohort_reads(manifest, reads)
    return _overlap_report(
        splits, reads, verified, relationship, {"evaluation_cohort_kind": "generated"},
    )


def build_registered_report(
    data_dir: Path, registry: Path, label: str, relationship: str
) -> Dict[str, Any]:
    """The generated `train` split against a registered cohort version.

    In order, refusing at the first failure and before anything is written:
    `train` is resolved; the version is resolved (`resolve_version`, one read of
    its ``cohort.json``); the workspace manifest is read once; the version's
    four recorded graph digests and its dataset record digest are compared with
    the ones that manifest records (`verify_cohort_binding`, the measurement's
    comparison); `train` is read once and verified against the same manifest
    reading; the version's ``samples.json`` is read once and refused before it
    is parsed unless its digest is the one ``cohort.json`` binds.

    **The graph is compared as recorded, and its files are not read.** This
    audit compares disease indices, not embeddings: what it must establish is
    that both sides index the same graph. The manifest that `train` is verified
    against records which graph that is, and the version records which graph it
    was mapped against, so the two records are what is compared. A version
    mapped against another graph is refused before its samples are opened.

    **Generated val is not read.** It is not an input to a train-versus-test
    overlap, so a missing or corrupt one must not block this measurement.
    """
    from src.evaluation.cohort_registry import resolve_version, verify_cohort_binding
    from src.kg.artifacts import (
        DATASET_RECORD_ROLE,
        GRAPH_ARTIFACTS,
        manifest_section,
        read_split_manifest,
    )
    from src.kg.storage.file_storage import read_samples, read_samples_file

    resolve_cohort(data_dir, TRAINING_SPLIT)
    version = resolve_version(registry, label)
    manifest = read_split_manifest(data_dir)
    artifacts = manifest_section(manifest.manifest, manifest.identity.path, "artifacts")
    verify_cohort_binding(
        version.manifest,
        {role: artifacts.get(role) for role in GRAPH_ARTIFACTS},
        artifacts.get(DATASET_RECORD_ROLE),
    )
    reads = {TRAINING_SPLIT: read_samples(data_dir, TRAINING_SPLIT)}
    verified = verify_cohort_reads(manifest, reads)
    # The role, not the label, names the evaluation side, as a measurement
    # manifest's `split` does; `validate_role` (inside `resolve_version`) has
    # already refused `train` and `val`, so it cannot collide with `train`.
    role = version.manifest["role"]
    reads[role] = read_samples_file(
        version.samples_path, expected_sha256=version.manifest["samples"]["sha256"]
    )
    cohort_version = {
        "label": version.manifest["label"],
        "role": role,
        "manifest_sha256": version.identity.sha256,
        "samples_sha256": reads[role].identity.sha256,
        "verified_against": [
            "the samples file's SHA-256, against the version manifest's samples digest",
            "the version's four graph digests and its dataset record digest, against "
            "the ones this workspace's manifest records",
        ],
    }
    return _overlap_report(
        [TRAINING_SPLIT, role], reads, verified, relationship,
        {"cohort_version": cohort_version}, cohort_label=label,
    )


def _overlap_report(
    splits: List[str], reads: Dict[str, Any], verified: Any, relationship: str,
    cohort: Dict[str, Any], cohort_label: Optional[str] = None,
) -> Dict[str, Any]:
    """The report over two verified reads. `cohort` says what the evaluation
    side is: the generated kind, or the registered version it was read from."""
    train_split, eval_split = splits
    train_ids = disease_ids(reads[train_split])
    eval_ids = disease_ids(reads[eval_split])
    train_set, eval_set = set(train_ids), set(eval_ids)

    # **An empty split cannot establish the fact this file exists for.** M4 bounds
    # what an evaluation metric can be evidence of, by measuring how much of the
    # evaluation cohort's disease set the training split already contains. With no
    # evaluation diseases the ratio has no denominator; with no training diseases
    # the overlap is zero for a reason that is about the workspace, and either
    # would be read as "no contamination" by someone citing the file later. The
    # shared sample reader is left alone — a split that exists and is empty is a
    # legitimate thing for it to return, and only this audit needs to refuse it.
    eval_name = (f"the cohort version {cohort_label!r}" if cohort_label is not None
                 else f"the {eval_split} split")
    for name, ids in ((f"the {train_split} split", train_ids), (eval_name, eval_ids)):
        if not ids:
            raise ValueError(
                f"{name} holds no samples. M4 compares two disease sets, and an "
                "overlap involving an empty one says nothing about contamination — "
                "only that there is nothing on that side to compare."
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
        "digests": {f"{split}_samples": reads[split].identity.sha256 for split in splits},
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
            verified, splits, {train_split: train_set, eval_split: eval_set},
            cohort_label=cohort_label,
        ),
        **cohort,
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
    parser.add_argument("--splits", nargs=2, default=None,
                        metavar=("TRAINING", "EVALUATION"),
                        help="Which two of the generator's splits to compare, training "
                             "first. Defaults to train and val, which is the pair the "
                             "current trainer selects checkpoints on. A test cohort is "
                             "not a split of the workspace: audit it with --registry "
                             "and --cohort instead.")
    parser.add_argument("--registry", type=Path, default=None,
                        help="The cohort registry (an absolute path outside the "
                             "repository) holding the version --cohort names. Given "
                             "with --cohort, never alone.")
    parser.add_argument("--cohort", default=None, metavar="LABEL",
                        help="A registered cohort version's label "
                             "(scripts/import_cohort.py). Its overlap with the "
                             "generated train split is measured, never refused: the "
                             "cohort was not cut from this disease universe, so the "
                             "overlap is the finding. The version must have been "
                             "mapped against this workspace's graph.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing --output. Off by default.")
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

    if (args.registry is None) != (args.cohort is None):
        raise SystemExit(
            "--registry and --cohort go together: --cohort names a registered cohort "
            "version and --registry the directory that keeps it. Nothing was written."
        )
    if args.cohort is not None and args.splits is not None:
        raise SystemExit(
            "--splits names two of the generator's splits; with --cohort the evaluation "
            f"side is the registered version and it is compared with {TRAINING_SPLIT}. "
            "Pass one or the other. Nothing was written."
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists. Pass --overwrite or write elsewhere.")

    # **Verification failures refuse before an artifact exists.** Earlier this
    # audit wrote its report and then refused, so a failure left something to
    # cite. That was right while this was the only gate; the same verifier now
    # runs at training and measurement, where writing a report makes no sense, and
    # a workspace that fails it is one to rebuild rather than to publish a
    # measurement about. The refusal message names which check failed on which
    # file, which is the citable content.
    try:
        if args.cohort is None:
            report = build_report(args.data_dir, list(args.splits or DEFAULT_SPLITS),
                                  args.deployment_relationship)
        else:
            report = build_registered_report(args.data_dir, args.registry, args.cohort,
                                             args.deployment_relationship)
    except OSError as exc:
        raise SystemExit(f"{exc}. Nothing was written.") from exc
    except ValueError as exc:
        raise SystemExit(f"{exc} Nothing was written.") from exc

    # Published whole (`publish_bytes`: staged beside the target, flushed and
    # renamed over it), so an existing report replaced under --overwrite is
    # either the old file or the new one, never a part of either.
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        digest = publish_bytes(args.output, data)
    except OSError as exc:
        raise SystemExit(
            f"{args.output} could not be written ({type(exc).__name__}: {exc}); "
            "no report was published, and a file already there was left as it was."
        ) from exc
    logger.info("%s -> %s (sha256 %s)", report["overlap"]["as_written"], args.output, digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
