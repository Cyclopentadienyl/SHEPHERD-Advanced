"""
Evaluation records live beside the checkpoint, not inside it.
=============================================================
`EVALUATION_COHORTS.md` §6.5. Writing a result into the `.pt` changes its
SHA-256, and the M1–M5 chain cites checkpoints *by* digest — so recording a
number inside the weights would invalidate every citation of the weights the
number is about.

The alternative is a ledger keyed by the digest, which is what this is: a JSON
file beside the checkpoints holding one record per (checkpoint, cohort, mode,
tie policy). The digest stays stable, and a reader can ask what results exist for
a set of weights without knowing where anyone happened to point `--output`.

**What an absent record means, exactly.** It does not establish that a checkpoint
was never evaluated. This provenance system is not closed and an evaluation can
happen outside it. What absence supports is the narrower, true statement: *this
ledger holds no result for these weights*. Every function here is written so that
distinction survives — nothing infers, nothing defaults, and nothing fills a gap
with a plausible value.

**Not a registry, and deliberately not becoming one.** No parent/child graph, no
lineage walk, no run ids, no cross-file joins. A record points back at the
measurement artifact it came from by digest and stops there; the full manifest
lives in that artifact and is not copied.

**Contradiction is refused, not merged.** A key identifies one measurement. If
the same key arrives with different numbers, one of the two is wrong about what
it measured, and a ledger that silently kept the newer one would erase the
evidence that they disagreed. There is no override flag: correcting a bad record
means editing a readable JSON file, which leaves a diff, rather than passing a
flag that leaves nothing.

Standard library only — no torch, no I/O beyond one file.

Module: src/evaluation/sidecar.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Bumped when a record's shape changes in a way that makes an old file
#: unreadable under the new rules. A reader that finds a version it does not know
#: refuses rather than guessing which fields it can trust.
LEDGER_SCHEMA_VERSION = 1

#: The conventional file name, beside the checkpoints it describes.
LEDGER_FILENAME = "evaluations.json"

#: What makes two records the same measurement.
#:
#: ``mode`` is in the key and §6.5's wording predates it: Mode A and Mode C over
#: one checkpoint and one cohort are different measurements with different
#: candidate universes, and a key without it would report them as a contradiction.
#: ``canonical_tie_policy_version`` stands where §6.5 says "metric schema
#: version" — it is the version of how ranks become numbers, which is the part of
#: the metric schema that can change an answer.
KEY_FIELDS: Tuple[str, ...] = (
    "checkpoint_digest",
    "cohort_role",
    "cohort_digest",
    "mode",
    "canonical_tie_policy_version",
)


def empty_ledger() -> Dict[str, Any]:
    """A ledger holding nothing, which is a different thing from no ledger."""
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "what_this_is": (
            "evaluation results for the checkpoints in this directory, keyed by "
            "checkpoint digest. An absent record means this ledger holds no "
            "result for those weights, not that they were never evaluated"
        ),
        "records": [],
    }


def read_ledger(path: Path) -> Dict[str, Any]:
    """Load a ledger, or an empty one where no file exists.

    A missing file is not an error: the first record has to be appendable to a
    directory that has none. A file with an unknown ``schema_version`` **is** an
    error, because reading it under this version's rules would be a guess about
    which of its fields still mean what they say.
    """
    if not path.exists():
        return empty_ledger()
    ledger = json.loads(path.read_text())
    version = ledger.get("schema_version")
    if version != LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"{path} is ledger schema {version!r}; this code reads "
            f"{LEDGER_SCHEMA_VERSION}. Read it with the revision that wrote it "
            "rather than reinterpreting its fields under new rules."
        )
    if not isinstance(ledger.get("records"), list):
        raise ValueError(f"{path} has no records list")
    return ledger


def record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """The identity of a measurement, as a tuple.

    A tuple rather than a joined string: joining needs a separator, a separator
    needs escaping, and an unescaped one lets two different keys collide. Nothing
    is gained by it here — the key is only ever compared, never used as a
    filename or a dict key in the artifact.
    """
    missing = [field for field in KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record is missing key field(s): {', '.join(missing)}")
    return tuple(record[field] for field in KEY_FIELDS)


def build_record(report: Dict[str, Any], source_digest: Optional[str]) -> Dict[str, Any]:
    """One record from one measurement artifact as ``measure_scorer`` writes it.

    Reads the manifest rather than taking the fields as arguments, so a record
    cannot describe a run different from the one whose artifact it came from —
    the same reason ``build_loader_config``'s output is handed to both the
    dataloader and the manifest instead of being rebuilt.

    ``split_manifest`` is carried through when the measured workspace had one. Its
    absence is recorded as ``None`` here rather than by omitting the field,
    because at this point the question has been asked and answered: this cohort
    has no recorded allocation. That is a fact worth stating, and it is what
    separates a pre-allocation `val` number from a disease-disjoint one.
    """
    manifest = report["manifest"]
    digests = manifest["artifact_digests"]
    metrics = dict(report["authoritative_metrics"])
    return {
        "checkpoint_digest": digests["checkpoint"],
        "cohort_role": manifest["split"],
        "cohort_digest": digests["samples"],
        "mode": manifest["mode"],
        "canonical_tie_policy_version": manifest["canonical_tie_policy_version"],
        "allocation": {
            "split_manifest_digest": digests.get("split_manifest"),
            "why_this_is_here": (
                "a val metric under a disease-disjoint cut and one under a "
                "sample-level slice are different quantities; the sample digest "
                "does not distinguish them"
            ),
        },
        "metrics": metrics,
        "n_ranked": report["n_ranked"],
        "n_ground_truth_absent": report["n_ground_truth_absent"],
        "runtime": {
            "software_revision": manifest["software_revision"],
            "cuda_executed": manifest["cuda_executed"],
            "torch_version": manifest["torch_version"],
            "amp_enabled": manifest["amp_enabled"],
        },
        "source_artifact_digest": source_digest,
    }


def append_record(ledger: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """Add a record, refusing a key that already carries different numbers.

    Re-appending an identical record is a no-op rather than a duplicate row: a
    ledger rebuilt from the same artifacts twice should be the same ledger.

    A **contradiction** — one key, two metric sets — is refused with the differing
    names in the message. It means one of the two runs is wrong about what it
    measured, and that is a finding, not a merge conflict to resolve by taking the
    newer value.
    """
    key = record_key(record)
    for existing in ledger["records"]:
        if record_key(existing) != key:
            continue
        if existing == record:
            return ledger
        differing = sorted(
            name
            for name in set(existing["metrics"]) | set(record["metrics"])
            if existing["metrics"].get(name) != record["metrics"].get(name)
        )
        raise ValueError(
            f"this ledger already holds a record for checkpoint "
            f"{record['checkpoint_digest'][:12]}... on cohort "
            f"{record['cohort_digest'][:12]}... in mode {record['mode']}, and the "
            + (
                f"two disagree on: {', '.join(differing)}. "
                if differing
                else "two differ outside their metrics. "
            )
            + "One of them is wrong about what it measured. Nothing was written."
        )
    return {**ledger, "records": ledger["records"] + [record]}


def records_for(ledger: Dict[str, Any], checkpoint_digest: str) -> List[Dict[str, Any]]:
    """Every result this ledger holds for one set of weights.

    An empty list is the honest answer to "what has this checkpoint been measured
    on", scoped to this ledger. It is never an answer to "has it been evaluated".
    """
    return [
        record
        for record in ledger["records"]
        if record["checkpoint_digest"] == checkpoint_digest
    ]


def write_ledger(path: Path, ledger: Dict[str, Any]) -> None:
    """Write the ledger, replacing the old file only once the new one is complete.

    A partial write is the failure mode that matters here: an interrupted append
    would leave unparseable JSON where a directory's entire evaluation history
    used to be, and the records it destroyed are not recoverable from the
    checkpoints. Writing to a temporary file in the same directory and renaming
    it makes the replacement atomic on every platform this runs on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), prefix=path.name, suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(ledger, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


__all__ = [
    "KEY_FIELDS",
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA_VERSION",
    "append_record",
    "build_record",
    "empty_ledger",
    "read_ledger",
    "record_key",
    "records_for",
    "write_ledger",
]
