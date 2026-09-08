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

import hashlib
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

#: Manifest fields that can change a number, hashed into one semantics digest.
#:
#: **A false contradiction is worse than a missed one, and this list is chosen on
#: that asymmetry.** A missed contradiction leaves two records the reader can
#: compare; a false one *blocks a legitimate append*, so the rule is to include
#: anything that could move a metric. An earlier key of
#: (checkpoint, cohort, mode, tie policy) failed exactly this way: `batch_size` is
#: documented on the manifest as semantics rather than performance — Mode A's
#: candidate universe is the batch's subgraph — so two honest runs at different
#: batch sizes collided and the second was refused as a contradiction.
#:
#: Explicit rather than "every field", so that adding a non-semantic field later
#: does not churn every key and silence the contradiction check.
SEMANTIC_MANIFEST_FIELDS: Tuple[str, ...] = (
    "mode",
    "cohort_kind",
    "candidate_construction",
    "negative_sampling_strategy",
    "num_negative_samples",
    "subgraph_strategy",
    "subgraph_hops",
    "num_neighbors",
    "max_subgraph_nodes",
    "batch_size",
    "score_semantics",
    "model_construction",
    "legacy_truncation_k",
    "legacy_tie_policy",
    "canonical_tie_policy_version",
    "metric_schema_version",
    "software_revision",
    "device",
    "dtype",
    "amp_enabled",
    "amp_dtype",
    "deterministic_algorithms",
)

#: Artifact roles whose bytes change what was measured. The graph tensors and the
#: allocation are as much a part of the measurement as the checkpoint is.
SEMANTIC_ARTIFACT_ROLES: Tuple[str, ...] = (
    "node_features", "edge_indices", "num_nodes", "split_manifest",
)

#: What makes two records the same measurement.
#:
#: ``mode`` was in an earlier version of this key and is now inside the semantics
#: digest with everything else that can move a number. What remains outside it are
#: the two identities a reader looks a record up by: whose weights, and which
#: cohort.
KEY_FIELDS: Tuple[str, ...] = (
    "checkpoint_digest",
    "cohort_role",
    "cohort_digest",
    "measurement_semantics_digest",
)


def measurement_semantics_digest(manifest: Dict[str, Any]) -> str:
    """SHA-256 over everything about a run that could change its numbers.

    **Not the source artifact's digest.** That changes with a timestamp, a
    reordered dict or an added descriptive field, so using it as identity would
    make every re-run a new record and the contradiction check would never fire.
    This hashes a named list of semantic fields, so two runs that differ in
    nothing that matters produce the same digest and must agree.

    A field the manifest does not carry is hashed as ``null`` rather than skipped,
    so a manifest that dropped one cannot collide with a manifest carrying a value
    there. An explicitly-null field and an absent one do hash alike, which is
    correct for the fields that legitimately carry ``null`` — ``amp_dtype`` is
    ``None`` exactly when AMP is off — and is not a case a well-formed manifest
    produces otherwise.
    """
    artifacts = manifest.get("artifact_digests", {})
    payload = {
        field: manifest.get(field, None) for field in SEMANTIC_MANIFEST_FIELDS
    }
    payload["artifacts"] = {
        role: artifacts.get(role, None) for role in SEMANTIC_ARTIFACT_ROLES
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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

    The cohort's **kind** is derived from the roles the measurement recorded, not
    re-asserted here. A generated cohort carries a ``split_manifest`` role because
    ``resolve_cohort`` requires one; a supplied cohort has none because it was
    never cut from this project's disease universe. Those are the only two
    possibilities that reach a measurement artifact — a workspace built before the
    allocation step is refused at the measurement, not recorded with a null.
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
        "metric_schema_version": manifest["metric_schema_version"],
        "measurement_semantics_digest": measurement_semantics_digest(manifest),
        "cohort": {
            "kind": manifest["cohort_kind"],
            "split_manifest_digest": digests.get("split_manifest"),
            "why_this_is_here": (
                "a generated cohort is disease-disjoint by construction and its "
                "manifest says which cut produced it; a supplied cohort carries no "
                "allocation, and its overlap with training is a measurement. The "
                "sample digest alone distinguishes neither"
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
            f"{record['cohort_digest'][:12]}... under identical measurement "
            f"semantics ({record['measurement_semantics_digest'][:12]}...), and the "
            + (
                f"two disagree on: {', '.join(differing)}. "
                if differing
                else "two differ outside their metrics. "
            )
            + "One of them is wrong about what it measured. Nothing was written."
        )
    return {**ledger, "records": ledger["records"] + [record]}


def find_checkpoint(directory: Path, digest: str) -> Optional[Path]:
    """A file in this directory whose bytes are the ones the report measured.

    **Otherwise "beside the checkpoint" is not a fact.** A ledger written into a
    directory that does not hold the weights it describes is a record filed under
    the wrong address, and the reader's whole reason for looking there is gone.

    Candidates are hashed in sorted order and the search stops at the first match,
    so the full scan is paid only when there is no match — the case that ends in a
    refusal anyway. Several files with identical bytes are fine: the digest is the
    identity, not the name.
    """
    from src.utils.fingerprint import file_sha256

    for candidate in sorted(directory.glob("*.pt")):
        if candidate.is_file() and file_sha256(candidate) == digest:
            return candidate
    return None


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


def ledger_digest(path: Path) -> Optional[str]:
    """The bytes a ledger held when it was read, or ``None`` if there was no file."""
    from src.utils.fingerprint import file_sha256

    return file_sha256(path)


def write_ledger(
    path: Path, ledger: Dict[str, Any], expected_digest: Optional[str] = None
) -> None:
    """Write the ledger, replacing the old file only once the new one is complete.

    Two different failures, and this closes both without a locking framework.

    **A partial write** would leave unparseable JSON where a directory's entire
    evaluation history used to be, and those records are not recoverable from the
    checkpoints. Writing to a temporary file in the same directory and renaming it
    makes the replacement atomic on every platform this runs on.

    **A concurrent writer** is not a corruption but a silent loss: two processes
    read the same ledger, each appends its own record, and the second replace
    erases the first append with no trace. ``expected_digest`` — the bytes the
    caller read — turns that into a refusal. It is optimistic concurrency, not
    locking: no lock file, no timeout, no recovery path, five lines. The ledger
    remains **single-writer by design**; this detects a violation rather than
    supporting one, and if the institutional workflow ever appends concurrently
    that is when locking earns its place.
    """
    if expected_digest is not None and ledger_digest(path) != expected_digest:
        raise ValueError(
            f"{path} changed since it was read, so appending would erase whatever "
            "the other writer added. The ledger is single-writer by design: "
            "re-read it and append again."
        )
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
    "ledger_digest",
    "SEMANTIC_ARTIFACT_ROLES",
    "SEMANTIC_MANIFEST_FIELDS",
    "measurement_semantics_digest",
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA_VERSION",
    "append_record",
    "build_record",
    "empty_ledger",
    "find_checkpoint",
    "read_ledger",
    "record_key",
    "records_for",
    "write_ledger",
]
