"""
Evaluation records live beside the checkpoint, not inside it.
=============================================================
`EVALUATION_COHORTS.md` §6.5. Writing a result into the `.pt` changes its
SHA-256, and the M1–M5 chain cites checkpoints *by* digest — so recording a
number inside the weights would invalidate every citation of the weights the
number is about.

The alternative is a ledger keyed by the digest, which is what this is: a JSON
file beside the checkpoints holding one record per measurement of a checkpoint
on a registered test cohort version. The digest stays stable, and a reader can
ask what results exist for a set of weights without knowing where anyone
happened to keep the report.

**What an absent record means, exactly.** It does not establish that a checkpoint
was never evaluated. This provenance system is not closed and an evaluation can
happen outside it. What absence supports is the narrower, true statement: *this
ledger holds no result for these weights*. Every function here is written so that
distinction survives — nothing infers, nothing defaults, and nothing fills a gap
with a plausible value.

**What a record copies, and what it does not** (schema 3 and 4). A record
copies what the read-only view shows and what lets records be grouped and
compared: the condition group (the key), the scoring semantics and the numerical
regime, the cohort version's identity and aggregate counts, the metrics, the
relation states, R10 and the environment. The full report stays the source: the
record names it by digest, and the rest of the manifest, the rank file and
anything at case level stay there. Since schema 4 a record also names its run
(``run_id``): the run directory in the version's report store that keeps the
report. The report's bytes do not carry it, so the ledger records it.

- **No run paths.** The manifest's `checkpoint_path` and `data_dir` stay in the
  report: they say where one run happened to find its files, and a checkpoint
  measured again from another place is the same measurement.
- **R10's evidence paths are kept.** The `r10` block is copied as the report
  has it (`src/evaluation/r10.py`), so each scope's `evidence.path` -- the file
  on the measuring machine whose bytes the checkpoint records, when one was
  found -- is in the record, and so, for a search left incomplete, are the
  places that could not be read, in its `evidence.problem`. The read-only view
  strips both
  (`src/api/routes/evaluations.py`); `record_evaluation.py --show` prints the
  record as it is, on the operator's machine.
- **No case ids.** The cohort block is aggregates only (`COHORT_VERSION_FIELDS`).
- **No prose.** A wording change between revisions would make two records of
  identical results read as different, so the surfaces render the standing
  text from `src/evaluation/caveats.py` instead.

**Not a registry, and deliberately not becoming one.** No parent/child graph, no
lineage walk, no cross-file joins. The registration sequence orders the records
of this one file; nothing relates two ledgers.

**Every run is kept; a repeat is the same run's same report** (policy rev 16,
from K8-6r.2). A run is one real execution: its run directory in the cohort
version's report store, verified with its cohort version, its report's digest
and its checkpoint. The key is the run's condition group: it groups runs to show
and compare them and never refuses one. The writer looks for this run first,
among the records that name a run (`find_run`): the same run with the same
report bytes is ``already_registered`` and nothing is written; the same run with
other bytes is refused, since a run's report is sealed when it is published.
Any other verified run gets a record of its own, whether its values equal
another run's or differ, and whatever its bytes or times. A difference between
runs of one group is described (`differing_fields`), never refused.

**Records written before each run had its own record are read as written.** A
schema 3 file's records, and a schema 4 file's records before its run-identity
boundary, name no run ("not recorded"); one may stand for several reports
(``corroborating_reports``). They are never consulted to decide an outcome: a
run whose bytes such a record holds is registered as its own record, with a
note that the overlap is possible and not confirmed (`historical_overlap`).
What such a record lacks never refuses a verified run. Only what keeps this
write from completing safely refuses it: a ledger that does not parse, a record
whose registration sequence cannot be read, an unreadable boundary, or this
run's own identity record damaged. Nothing is guessed, skipped, deleted or
renumbered. There is no override flag: correcting a bad record means editing a
readable JSON file, which leaves a diff, rather than passing a flag that leaves
nothing.

Standard library only — no torch, no I/O beyond one file.

Module: src/evaluation/sidecar.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from src.evaluation.measurement import expected_relations
from src.evaluation.protocols import REGISTERED_KINDS, SUPPLIED
from src.utils.file_lock import exclusive_lock
from src.utils.file_modes import create_staging_file

#: Bumped when a record's meaning or the rules a writer must follow change, so
#: that a writer that does not know them refuses the file instead of applying old
#: rules to it, and when a record's shape makes an old file unreadable. The
#: reader keeps reading every version whose meaning it knows
#: (`READABLE_LEDGER_SCHEMAS`); one that finds a version it does not know refuses
#: rather than guessing which fields it can trust.
#:
#: **4 since K8-6r.2: each record names its run from the file's run-identity
#: boundary on; a repeat is the same run's same report.** A writer that knows
#: only schema 3 would still merge a report into the first record of its group;
#: it refuses a schema 4 file at read instead. An append writes schema 4 and
#: leaves every earlier record as written.
#:
#: **3 since M4.** A schema 3 record is keyed by the cohort version's manifest
#: digest rather than the samples digest, and carries what schema 2 lacked: the
#: relation states, the cohort's counts with their denominators, the scoring
#: semantics and regime, R10, the environment, a registration sequence and the
#: corroborating reports. A schema 2 record read under these rules would sit
#: beside them as though comparable with fields missing, so a schema 1 or 2 file
#: is refused and is read with the revision that wrote it. (2 was the Mode A/B
#: retirement's R2, which shortened the semantics field list.)
LEDGER_SCHEMA_VERSION = 4

#: The schemas the one reader reads, each with its own meaning: a schema 3 file
#: as written (every record "not recorded" for its run), a schema 4 file with its
#: boundary. Reading never rewrites a file.
READABLE_LEDGER_SCHEMAS: Tuple[int, ...] = (3, 4)

#: The schema 4 ledger field naming the registration sequence from which every
#: record names its run (`run_identity_boundary`). Set by the first append to a
#: file that has none; a record before it is historical, a record from it on
#: must carry a readable ``run_id``.
RUN_IDENTITY_FROM_SEQUENCE = "run_identity_from_sequence"

#: A run directory's name, as `cohort_registry.create_report_dir` makes it,
#: restated here (standard library only): importing `cohort_registry` would load
#: the graph package. A test pins it equal to `cohort_registry.RUN_ID_PATTERN`.
RUN_ID_PATTERN = re.compile(r"[0-9]{8}T[0-9]{6}Z-([0-9a-f]{12})-[0-9a-f]{8}")

#: The conventional file name, beside the checkpoints it describes.
LEDGER_FILENAME = "evaluations.json"

#: The data kinds whose runs this revision's ledger records (`build_record`):
#: a supplied cohort version's only. A run on another registered kind (a KG
#: edge set's link-prediction run) is kept in its version's report store and
#: not recorded yet; making the record per kind is research comparison §5
#: Stage 6. `scripts/measure_scorer.py` reads it so that it never offers a
#: registration this revision would refuse for certain.
LEDGER_KINDS: Tuple[str, ...] = (SUPPLIED,)

#: The ledger's lock file is the ledger's name with this added, beside it
#: (`ledger_lock_path`); it is never removed.
LEDGER_LOCK_SUFFIX = ".lock"

#: How long a registration waits for another one writing the same ledger
#: before it is refused as busy (`ledger_lock`), read at each call.
LEDGER_LOCK_GRACE_SECONDS = 10.0

#: Manifest fields that can change a number, hashed into one semantics digest.
#:
#: **Anything that could move a metric is included**, so that two runs in one
#: condition group differ only in what the recorded conditions do not fix, and a
#: difference between them is one to investigate rather than one a left-out
#: condition explains. An earlier key of (checkpoint, cohort, mode, tie policy)
#: put two honest runs at different batch sizes in one group, although
#: `batch_size` is the chunk of cases a run scores at once and can move its
#: numbers at floating-point level. The group only describes: since K8-6r.2 no
#: group refuses a run.
#:
#: Explicit rather than "every field", so that adding a non-semantic field later
#: does not split every group. That claim is
#: only worth making if it is checkable: this list and ``NON_SEMANTIC_FIELDS``
#: below must together cover every field of ``MeasurementManifest``, and
#: ``tests/unit/test_evaluation_sidecar.py`` fails if a new field belongs to
#: neither. A field cannot be omitted by oversight, only by decision.
SEMANTIC_MANIFEST_FIELDS: Tuple[str, ...] = (
    # what was ranked, against what, and in what chunks
    "mode",
    "cohort_kind",
    "candidate_construction",
    "batch_size",
    # how scores became numbers
    "score_semantics",
    "model_construction",
    "canonical_tie_policy_version",
    "metric_schema_version",
    # the numerical regime. `torch_compile_wrapped` is included although `None`
    # means "not observed" rather than "not compiled": two runs where one was
    # observed and one was not are not provably the same measurement, and the
    # asymmetry above says include.
    "software_revision",
    "torch_version",
    "cuda_version",
    "device",
    "dtype",
    "amp_enabled",
    "amp_dtype",
    "torch_compile_wrapped",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "cudnn_benchmark",
)

#: D4's scoring semantics: what makes two results rankings of the same kind, so
#: records of different checkpoints can be grouped by these. A record carries
#: them and their own digest (`scoring_semantics_digest`).
SCORING_SEMANTICS_FIELDS: Tuple[str, ...] = (
    "mode",
    "cohort_kind",
    "candidate_construction",
    "score_semantics",
    "model_construction",
    "canonical_tie_policy_version",
    "metric_schema_version",
)

#: The rest of the semantic list: the regime a number was computed under. Two
#: records in one scoring group can differ here, and each record keeps it, so
#: that difference can be shown rather than hidden. With ``SCORING_SEMANTICS_FIELDS``
#: it partitions ``SEMANTIC_MANIFEST_FIELDS``, which a test checks.
REGIME_FIELDS: Tuple[str, ...] = (
    "batch_size",
    "software_revision",
    "torch_version",
    "cuda_version",
    "device",
    "dtype",
    "amp_enabled",
    "amp_dtype",
    "torch_compile_wrapped",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "cudnn_benchmark",
)

#: Manifest fields deliberately outside the digest, each for a stated reason.
#:
#: Present so that "the list covers everything that can move a metric" is a
#: checkable claim rather than an assertion. Adding a field to
#: ``MeasurementManifest`` without deciding which side it falls on fails a test.
NON_SEMANTIC_FIELDS: Dict[str, str] = {
    "split": "the cohort role, already a key field in its own right",
    "n_samples": (
        "derived from the cohort, which the cohort digest already identifies; it "
        "cannot differ between two runs over the same cohort digest"
    ),
    "checkpoint_path": "a path is not an identity; the checkpoint digest is",
    "data_dir": "a path is not an identity; the artifact digests are",
    "graph_fingerprint": (
        "structural identity — node types, counts, feature dims. The graph "
        "artifact digests below are strictly stronger: identical bytes imply "
        "identical structure, and two graphs can share a fingerprint and differ"
    ),
    "artifact_digests": (
        "hashed selectively rather than wholesale: the checkpoint is a key field "
        "and the samples are bound by the cohort digest, the graph tensors and the "
        "allocation are covered by SEMANTIC_ARTIFACT_ROLES, and `kg` is the "
        "manifest-bound source graph, which no mode computes from"
    ),
    "cuda_executed": (
        "implied by `device`, which is in the digest; `_resolve_device` sets the "
        "two together, so it cannot vary independently"
    ),
    "dataset_record": (
        "what the graph was built from and whether the record is bound to it "
        "(contract M3b). No mode computes from it, and the graph tensors it "
        "describes are SEMANTIC_ARTIFACT_ROLES; its digest is bound by the "
        "split manifest"
    ),
    "cohort_version": (
        "which registered cohort version was scored. Its manifest digest is the "
        "record's cohort digest, a key field in its own right, and that manifest "
        "binds everything else in this block"
    ),
    "relations": (
        "the state of each relation on the bytes read. A written manifest's states "
        "are fixed by the kind of cohort, since a relation that does not hold "
        "refuses the run, so they cannot differ between two runs over one cohort"
    ),
    "r10": (
        "the cohort's overlap with the training and validation inputs, from the "
        "evidence files found. A re-run may find evidence the first did not, so "
        "keying it would split one measurement in two; it is recorded with each "
        "result and true of its own report"
    ),
    "environment": (
        "the contract §5.3 facts, recorded and never refused (R11). The facts that "
        "can move a number are semantic fields above; the GPU, driver, cuDNN, TF32 "
        "and PyG facts inform an investigation of two runs whose results differ"
    ),
}

#: Artifact roles whose bytes change what was measured. The graph tensors and the
#: allocation are as much a part of the measurement as the checkpoint is.
SEMANTIC_ARTIFACT_ROLES: Tuple[str, ...] = (
    "node_features", "edge_indices", "num_nodes", "split_manifest",
)

#: A record's condition group: what makes two runs measurements under the same
#: recorded conditions, to show and compare them. Never a run's identity (a run
#: is its ``run_id``), and never a reason to refuse one.
#:
#: ``mode`` was in an earlier version of this key and is now inside the semantics
#: digest with everything else that can move a number. What remains outside it are
#: the two identities a reader looks a record up by — whose weights, and which
#: cohort — and the cohort's role. Since schema 3 the cohort is the **cohort
#: version's manifest digest**: it binds the samples, the source, the graph, the
#: mapping policy and the counts, so two versions with identical samples but
#: different sources are two cohorts.
KEY_FIELDS: Tuple[str, ...] = (
    "checkpoint_digest",
    "cohort_role",
    "cohort_digest",
    "measurement_semantics_digest",
)

#: The relation states a registered result carries: R1, R3, R5 and the cohort's
#: binding to the graph verified, and R2, a generated split's relation, not
#: applicable. Taken from `src/evaluation/measurement.py`, which states them once.
REGISTERED_RELATIONS: Dict[str, str] = expected_relations(registered=True)

#: What a registered report's ``cohort_version`` block holds, exactly. Aggregates
#: only: a block naming anything else is refused, so a case id cannot ride into a
#: record on a field nobody classified.
COHORT_VERSION_FIELDS: Tuple[str, ...] = (
    "label",
    "role",
    "manifest_sha256",
    "source_sha256",
    "source_format",
    "samples_sha256",
    "graph",
    "dataset_record_sha256",
    "mapping",
    "counts",
)

#: The parts of the cohort version a record copies into its ``cohort`` block,
#: beside the cohort kind. The manifest digest is the key's ``cohort_digest``, and
#: the dataset record's digest is the record's own ``dataset_record``.
_RECORD_COHORT_FIELDS: Tuple[str, ...] = (
    "label",
    "role",
    "samples_sha256",
    "source_sha256",
    "source_format",
    "graph",
    "mapping",
    "counts",
)

#: Record fields compared between two runs of one condition group to describe how
#: they differ (`differing_fields`): the report store's description of a group and
#: the registration's group note. A difference decides nothing.
RECORD_COMPARED_FIELDS: Tuple[str, ...] = (
    "scoring_semantics_digest",
    "scoring",
    "regime",
    "cohort",
    "metrics",
    "n_ranked",
    "relations",
    "dataset_record",
)

#: Record fields outside the comparison, each for a stated reason. With
#: ``KEY_FIELDS`` and ``RECORD_COMPARED_FIELDS`` they cover every field of a
#: registered record, which a test checks, so a field cannot escape the
#: comparison by oversight.
RECORD_EXCLUDED_FIELDS: Dict[str, str] = {
    "r10": (
        "a re-run may find evidence the first did not, so two runs of one "
        "measurement can carry different labels; each record's labels are true of "
        "its own report"
    ),
    "environment": (
        "R11: recorded, never refused. A difference informs an investigation of "
        "two runs; each record keeps its own run's"
    ),
    "source_artifact_digest": (
        "the report's bytes, which can differ between two runs under one group "
        "(their paths, their environment) and can also be identical (two runs on "
        "one machine). With the run it tells a repeat (the same run's same bytes) "
        "from a replaced report; it never stands in for the run"
    ),
    "run_id": (
        "the run the record was registered from (schema 4), assigned by the ledger "
        "(`append_record`): it tells runs apart and is never compared"
    ),
    "corroborating_reports": (
        "reports merged into a record before each run had its own record (schema 3, "
        "and records before a schema 4 file's run-identity boundary); [] on every "
        "record registered since"
    ),
    "registration": "when and in what order this ledger registered the record",
}

#: What ``append_record`` can do with a record, by name. (``corroborated``, the
#: merge of another run's report into a group's first record, retired in
#: K8-6r.2; a usage-history entry written with it is read as written.)
REGISTRATION_OUTCOMES: Tuple[str, ...] = ("registered", "already_registered")

#: How a record's regime is described where a person reads it: the device class
#: and whether deterministic algorithms were on, which says whether two runs of
#: one group whose values differ were expected to agree (on: investigate).
ENVIRONMENT_CLASSES: Tuple[str, ...] = (
    "CUDA, deterministic algorithms on",
    "CUDA, deterministic algorithms off",
    "CPU (development; a different measurement from a CUDA run)",
)


def _canonical_digest(payload: Dict[str, Any]) -> str:
    """SHA-256 of one canonical JSON form of a value in memory.

    One form for every digest this module takes over manifest fields, so the
    semantics digest and the scoring-semantics digest cannot drift into two
    encodings of the same values: keys sorted, compact separators, and `str` for
    anything JSON has no form for.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def measurement_semantics_digest(manifest: Dict[str, Any]) -> str:
    """SHA-256 over everything about a run that could change its numbers.

    **Not the source artifact's digest.** That changes with a timestamp, a
    reordered dict or an added descriptive field, so grouping by it would put
    every run in a group of its own, and runs under identical conditions could
    not be compared. This hashes a named list of semantic fields, so two runs
    that differ in nothing that matters produce the same digest.

    **What "the same conditions" means here is narrower than bit-determinism.**
    Two records under one digest were produced under identical recorded
    semantics. Whether they were therefore *expected* to agree depends on the
    regime those fields describe: under `deterministic_algorithms` a difference
    asks for an investigation, and under a non-deterministic CUDA regime it may
    be variation the regime permits. Either way each run keeps its own record,
    and the difference is described, never refused.

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
    return _canonical_digest(payload)


def scoring_semantics_digest(manifest: Dict[str, Any]) -> str:
    """SHA-256 over D4's seven scoring fields: which results are rankings of the
    same kind and may be grouped for a comparison.

    Coarser than ``measurement_semantics_digest`` on purpose: two runs at
    different batch sizes or on different revisions are two measurements, and
    still the same scoring, so they can be grouped with their regime difference
    shown. Absent fields hash as ``null``, as there.
    """
    return _canonical_digest(
        {field: manifest.get(field, None) for field in SCORING_SEMANTICS_FIELDS}
    )


def empty_ledger() -> Dict[str, Any]:
    """A ledger holding nothing, which is a different thing from no ledger.

    Schema 4 with no records and no run-identity boundary: no record's identity
    is undetermined and the next sequence is plainly 1, so its boundary is 1
    (`run_identity_boundary`) and the first append writes it."""
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "what_this_is": (
            "evaluation results for the checkpoints in this directory, keyed by "
            "checkpoint digest. An absent record means this ledger holds no "
            "result for those weights, not that they were never evaluated"
        ),
        "records": [],
    }


def read_ledger(path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load a ledger and the digest of the bytes parsed, or an empty one and
    ``None`` where no file exists.

    **One read** (BACKLOG item 18): the ledger is parsed from the bytes whose
    digest is returned, so a writer's stale-write guard compares the file with
    what it actually parsed. Reading the file and hashing it again let another
    writer's append land in between and be erased by a write that passed the
    guard.

    A missing file is not an error: the first record has to be appendable to a
    directory that has none, and `empty_ledger` is returned for it. A file that
    is not UTF-8 JSON, not an object, has no records list or carries a
    ``schema_version`` outside `READABLE_LEDGER_SCHEMAS` **is** an error, named
    by its path, because reading it under this version's rules would be a guess
    about which of its fields still mean what they say.

    **Schema 3 and 4 are read as they are, and nothing is rewritten** (K8-6r.2):
    a schema 3 file's records name no run; a schema 4 file's boundary is read by
    `run_identity_boundary`, not here, so a file whose boundary is lost still
    reads, and its readable results stay readable. Only an append writes schema
    4 (`append_record`).
    """
    from src.utils.fingerprint import read_once

    path = Path(path)
    try:
        read = read_once(path)
    except FileNotFoundError:
        return empty_ledger(), None
    digest = read.sha256
    try:
        text = read.data.decode("utf-8")
        del read
        ledger = json.loads(text)
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        raise ValueError(
            f"{path} is not a readable ledger: it does not parse as UTF-8 JSON "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(ledger, dict):
        raise ValueError(
            f"{path} is not a ledger: it holds a JSON {type(ledger).__name__}, not an "
            "object"
        )
    version = ledger.get("schema_version")
    readable = " and ".join(str(v) for v in READABLE_LEDGER_SCHEMAS)
    if type(version) is int and version in (1, 2):
        # **Not "register again".** No report written before schema 3 can be
        # registered under it (`build_record` refuses its manifest), and this text
        # is shown where no current report is at hand (the model status,
        # `--show`), so it says what can be done: re-measure.
        raise ValueError(
            f"{path} is ledger schema {version}; this code reads schemas "
            f"{readable}. Archive it, for example as "
            f"evaluations.v{version}.json; it is read with the revision that wrote "
            "it. Its results are not carried over: re-measure with this revision "
            "and register the new reports."
        )
    if type(version) is not int or version not in READABLE_LEDGER_SCHEMAS:
        raise ValueError(
            f"{path} is ledger schema {version!r}; this code reads schemas "
            f"{readable}. Read it with the revision that wrote it "
            "rather than reinterpreting its fields under new rules."
        )
    if not isinstance(ledger.get("records"), list):
        raise ValueError(f"{path} has no records list")
    return ledger, digest


def record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """A record's condition group (`KEY_FIELDS`), as a tuple: what its runs are
    grouped and compared by, never a run's identity.

    A tuple rather than a joined string: joining needs a separator, a separator
    needs escaping, and an unescaped one lets two different keys collide. Nothing
    is gained by it here — the key is only ever compared, never used as a
    filename or a dict key in the artifact.
    """
    missing = [field for field in KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record is missing key field(s): {', '.join(missing)}")
    return tuple(record[field] for field in KEY_FIELDS)


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


#: How deep a report may nest, counting each object and list as a level. The
#: measurement writes 7 or 8; anything near this is a hand edit. Counted, not
#: recursed, so the refusal never depends on the interpreter's recursion limit
#: or on how deep the caller's stack already is.
MAX_REPORT_DEPTH = 64


class ReportNestsTooDeeply(RecursionError):
    """A report nested deeper than `MAX_REPORT_DEPTH`. A `RecursionError`, so the
    callers that refuse one (`register_report`, the report store's scan) read it
    the same way as a copy that ran out of stack."""


def _nests_deeper_than(value: Any, limit: int) -> bool:
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return True
        if isinstance(node, dict):
            stack.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            stack.extend((child, depth + 1) for child in node)
    return False


def _mapping(value: Any, what: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not an object")
    return value


def build_record(report: Dict[str, Any], source_digest: str) -> Dict[str, Any]:
    """One record from one measurement report of a registered cohort version.

    Reads the manifest rather than taking the fields as arguments, so a record
    cannot describe a run different from the one whose report it came from.
    ``source_digest`` is required: it is the digest of the bytes the report was
    parsed from, taken by the same read (`register_report`), and the record names
    the report by it.

    **Only a registered cohort version's report is recorded.** A generated split
    is refused because the ledger does not yet support a generated split's data
    identity: it records results by registered cohort version (D3 labels a
    result by role *and version*), and a generated split has none. That is a
    limit of the ledger, not a judgement that such a result counts for less
    (policy rev 16): its report and ranks stay at `--output`, and extending the
    ledger to them is later work. A run on a registered version of a kind
    outside `LEDGER_KINDS` (a KG edge set's link-prediction run, research
    comparison Stage 5) is refused as one this revision's ledger does not
    record yet; its report stays in its version's report store. A report whose relation states are not a registered result's is refused too;
    the measurement cannot write one, so such a file was edited or is not a
    report. Every refusal is a ``ValueError`` naming what is wrong — never a
    ``KeyError`` or ``TypeError`` from a file of the wrong shape.

    Returned without ``corroborating_reports`` and ``registration``, which are
    the ledger's to assign (`append_record`).
    """
    if not _is_digest(source_digest):
        raise ValueError(
            f"the report's digest is {source_digest!r}, not a SHA-256: a record names "
            "the bytes it was built from"
        )
    if not isinstance(report, dict):
        raise ValueError(
            f"this is a JSON {type(report).__name__}, not a measurement report (an "
            "object with a manifest)"
        )
    if _nests_deeper_than(report, MAX_REPORT_DEPTH):
        raise ReportNestsTooDeeply(
            f"it nests deeper than {MAX_REPORT_DEPTH} levels, which no measurement writes"
        )
    manifest = report.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            "it has no manifest object, so it is not a measurement report this can "
            "derive a record from"
        )
    # **Only the current measurement's manifest is read.** A field outside the two
    # lists above is one this revision does not record — written by a retired mode
    # or before the field was removed — and hashing such a manifest under the
    # current list would drop what told its runs apart. A manifest lacking one of
    # this revision's fields was written before it existed. Both are refused.
    known = set(SEMANTIC_MANIFEST_FIELDS) | set(NON_SEMANTIC_FIELDS)
    unknown = sorted(set(manifest) - known)
    missing = sorted(known - set(manifest))
    if unknown or missing:
        raise ValueError(
            "this measurement's manifest "
            + (f"records {', '.join(unknown)}, which the current measurement does "
               "not" if unknown else "")
            + ("; it " if unknown and missing else "")
            + (f"lacks {', '.join(missing)}, which the current measurement records"
               if missing else "")
            + ". It was written by an earlier revision or a retired mode, and is not "
            "read. Re-measure with this revision."
        )
    cohort_version = manifest["cohort_version"]
    if cohort_version is None:
        raise ValueError(
            f"this is a generated split's report (`{manifest['split']}`, no cohort "
            "version), and a generated split's report is not registered: the ledger "
            "does not yet support a generated split's data identity (it has no "
            "registered cohort version), not because such a result counts for less; "
            "its report and ranks stay at --output. To record a result in the ledger, "
            "measure a registered version (--cohort LABEL --registry DIR)."
        )
    kind = manifest["cohort_kind"]
    if (kind in REGISTERED_KINDS and kind not in LEDGER_KINDS and isinstance(cohort_version, dict)
            and cohort_version.get("kind") == kind):
        # A run on a registered version of another kind than a supplied
        # cohort (research comparison Stage 5): kept in its version's report
        # store, and not yet recordable here. Said as this revision's gap,
        # never as a report that contradicts itself.
        raise ValueError(
            f"this report measured a registered version of kind {kind!r}, and this "
            "revision's ledger records runs on a supplied cohort version only (a record "
            "of another kind has fields it does not yet carry), so it is not recorded; "
            "the report is kept in its version's report store as written"
        )
    if kind != SUPPLIED:
        raise ValueError(
            f"this report records cohort_kind {manifest['cohort_kind']!r} beside a "
            "cohort version; a registered cohort version is a supplied cohort, so "
            "the report contradicts itself and is not recorded"
        )
    cohort_version = _mapping(cohort_version, "its cohort_version")
    cv_missing = [f for f in COHORT_VERSION_FIELDS if f not in cohort_version]
    cv_unknown = sorted(set(cohort_version) - set(COHORT_VERSION_FIELDS))
    if cv_missing or cv_unknown:
        raise ValueError(
            "its cohort_version "
            + (f"lacks {', '.join(cv_missing)}" if cv_missing else "")
            + ("; it " if cv_missing and cv_unknown else "")
            + (f"names {', '.join(cv_unknown)}, which a cohort version's report "
               "block does not carry" if cv_unknown else "")
            + ", so it is not a cohort version this revision records"
        )
    for field in ("label", "role", "manifest_sha256", "source_sha256", "source_format",
                  "samples_sha256", "dataset_record_sha256"):
        if not isinstance(cohort_version[field], str) or not cohort_version[field]:
            raise ValueError(f"its cohort_version.{field} is not a non-empty string")
    for field in ("graph", "mapping", "counts"):
        _mapping(cohort_version[field], f"its cohort_version.{field}")
    if manifest["split"] != cohort_version["role"]:
        raise ValueError(
            f"its manifest names the cohort role {manifest['split']!r} and its cohort "
            f"version the role {cohort_version['role']!r}; a record is filed under "
            "one role, so the report contradicts itself and is not recorded"
        )
    relations = _mapping(manifest["relations"], "its relations")
    if relations != REGISTERED_RELATIONS:
        wrong = sorted(
            f"{name} {relations.get(name, 'absent')!r}"
            for name in set(relations) | set(REGISTERED_RELATIONS)
            if relations.get(name) != REGISTERED_RELATIONS.get(name)
        )
        raise ValueError(
            f"its relation states are not a registered result's ({', '.join(wrong)}): "
            "R1, R3, R5 and cohort_graph are verified and R2 does not apply on every "
            "report the measurement writes for a cohort version, so this one is not "
            "recorded"
        )
    digests = _mapping(manifest["artifact_digests"], "its artifact_digests")
    if not _is_digest(digests.get("checkpoint")):
        raise ValueError("its artifact_digests name no checkpoint SHA-256")
    dataset_record = _mapping(manifest["dataset_record"], "its dataset_record")
    for field in ("origin", "sha256"):
        if not isinstance(dataset_record.get(field), str):
            raise ValueError(f"its dataset_record.{field} is not a string")
    r10 = _mapping(manifest["r10"], "its r10 block")
    environment = _mapping(manifest["environment"], "its environment")
    metrics = _mapping(report.get("authoritative_metrics"), "its authoritative_metrics")
    n_ranked = report.get("n_ranked")
    if type(n_ranked) is not int:
        raise ValueError(f"its n_ranked is {n_ranked!r}, not an integer")

    return {
        "checkpoint_digest": digests["checkpoint"],
        "cohort_role": manifest["split"],
        "cohort_digest": cohort_version["manifest_sha256"],
        "measurement_semantics_digest": measurement_semantics_digest(manifest),
        "scoring_semantics_digest": scoring_semantics_digest(manifest),
        "scoring": {field: manifest[field] for field in SCORING_SEMANTICS_FIELDS},
        "regime": {field: manifest[field] for field in REGIME_FIELDS},
        "cohort": {
            "kind": manifest["cohort_kind"],
            **{field: copy.deepcopy(cohort_version[field])
               for field in _RECORD_COHORT_FIELDS},
        },
        "metrics": copy.deepcopy(metrics),
        "n_ranked": n_ranked,
        "relations": dict(relations),
        "dataset_record": {
            "origin": dataset_record["origin"], "sha256": dataset_record["sha256"],
        },
        "r10": copy.deepcopy(r10),
        "environment": copy.deepcopy(environment),
        "source_artifact_digest": source_digest,
    }


def _readable_sequence(record: Any) -> Optional[int]:
    """A record's registration sequence, or None when it cannot be read."""
    registration = record.get("registration") if isinstance(record, dict) else None
    sequence = registration.get("sequence") if isinstance(registration, dict) else None
    return sequence if type(sequence) is int and sequence >= 1 else None


def sequence_problem(ledger: Dict[str, Any]) -> Optional[str]:
    """Why the next registration sequence cannot be allocated safely, naming
    the first record, in file order, that is not an object or has no readable
    registration sequence; None when every record's sequence reads. Refused
    rather than guessed: a new sequence computed past a gap could repeat one.

    **What makes a write itself uncertain** (K8-6r.2 (ii)(b)0), in one place:
    `append_record` refuses an append with it, and `report_state` reads
    ``cannot_be_checked`` with it when no record names the run, so the state
    and the writer agree."""
    for index, record in enumerate(_ledger_records(ledger)):
        if not isinstance(record, dict):
            return (f"the ledger's record {index} is not an object, so the next registration "
                    "sequence cannot be allocated safely")
        if _readable_sequence(record) is None:
            return (f"the ledger's record {index} has no registration sequence, so the next "
                    "sequence cannot be allocated safely; it was edited into a shape this code "
                    "cannot append to without guessing")
    return None


def _run_id(record: Any) -> Optional[str]:
    """The run a record names, when it names one readably: a run directory's name
    (`RUN_ID_PATTERN`). None when the field is missing or is anything else."""
    run_id = record.get("run_id") if isinstance(record, dict) else None
    return run_id if isinstance(run_id, str) and RUN_ID_PATTERN.fullmatch(run_id) else None


def _ledger_records(ledger: Dict[str, Any]) -> List[Any]:
    records = ledger.get("records") if isinstance(ledger, dict) else None
    return records if isinstance(records, list) else []


def _schema_4(ledger: Dict[str, Any]) -> bool:
    version = ledger.get("schema_version") if isinstance(ledger, dict) else None
    return type(version) is int and version == 4


def run_identity_boundary(ledger: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """The sequence from which every record of this ledger names its run, as
    ``(boundary, None)``; ``(None, why)`` when it cannot be read; ``(None, None)``
    for a schema 3 file, which has none (every record is historical).

    **The one place the boundary is read** (K8-6r.2): `find_run`,
    `append_record`, `historical_overlap` and the readers' record view
    (`record_identities`) all take it from here, so they agree. A schema 4 file
    that holds no record and names no boundary is not damaged: its boundary is 1,
    and the first append writes it. Unreadable: missing while the file holds
    records, not an integer, below 1, or above the next sequence. Reading it
    never changes the file.
    """
    if not _schema_4(ledger):
        return None, None
    records = _ledger_records(ledger)
    if RUN_IDENTITY_FROM_SEQUENCE not in ledger:
        if not records:
            return 1, None
        return None, (f"it is missing while the file holds {len(records)} "
                      f"record{'' if len(records) == 1 else 's'}")
    value = ledger[RUN_IDENTITY_FROM_SEQUENCE]
    if type(value) is not int:
        # Not echoed: every reader lists a record with this reason (K8-6e.1).
        return None, "it is not an integer"
    if value < 1:
        return None, f"it is {value}, below 1"
    following = max((s for s in map(_readable_sequence, records) if s is not None),
                    default=0) + 1
    if value > following:
        return None, f"it is {value}, above the next sequence {following}"
    return value, None


def _boundary_lost(why: str) -> str:
    return (f"the file's run-identity boundary ({RUN_IDENTITY_FROM_SEQUENCE}) cannot be "
            f"read ({why})")


class RecordIdentity(NamedTuple):
    """How one ledger record's run reads (`record_identities`).

    - ``run_id`` set: a result naming its run.
    - ``historical``: written before each run had its own record (a schema 3
      file, or before a schema 4 file's boundary); its run is "not recorded".
    - ``problem`` set: its identity cannot be read (at or after the boundary
      without a readable ``run_id``, or the boundary lost); listed as an
      unreadable item with this reason, never shown as historical.
    """

    run_id: Optional[str]
    historical: bool
    problem: Optional[str]


def record_identities(ledger: Dict[str, Any]) -> List[RecordIdentity]:
    """Each record's identity, in file order, by the file's boundary
    (`run_identity_boundary`), **never by the presence of a ``run_id`` key
    alone**: a record before the boundary is historical whatever it carries, and
    a hand edit that removes a ``run_id`` cannot make a later record look
    historical, since its sequence still lies at or after the boundary.

    A record whose sequence cannot be read is a result when it names its run,
    and otherwise unreadable; with the boundary lost, likewise for every record.
    Reads only; for every reader and for `find_run`.
    """
    boundary, why = run_identity_boundary(ledger)
    out: List[RecordIdentity] = []
    for record in _ledger_records(ledger):
        if not isinstance(record, dict):
            out.append(RecordIdentity(None, False, "it is not an object"))
            continue
        if not _schema_4(ledger):
            out.append(RecordIdentity(None, True, None))
            continue
        run_id = _run_id(record)
        sequence = _readable_sequence(record)
        if boundary is not None and sequence is not None and sequence < boundary:
            out.append(RecordIdentity(None, True, None))
        elif run_id is not None:
            out.append(RecordIdentity(run_id, False, None))
        elif boundary is None:
            out.append(RecordIdentity(None, False, (
                f"whether it names its run cannot be told: {_boundary_lost(why)}")))
        elif sequence is None:
            out.append(RecordIdentity(None, False, (
                "it has no readable registration sequence, so whether it must name its "
                "run cannot be told")))
        elif "run_id" not in record:
            out.append(RecordIdentity(None, False, "its run identity is missing"))
        else:
            # The field is named, never its value: a reader lists the record
            # with these words (K8-6e.1), and a hand edit can put a server
            # path there.
            out.append(RecordIdentity(None, False, (
                "its run identity cannot be read: its run_id is not a run directory's name")))
    return out


class RunLookup(NamedTuple):
    """What `find_run` found: the record naming the run and its index, or
    neither; or why whether this run is registered cannot be told (``problem``),
    in which case nothing may be appended for it."""

    index: Optional[int]
    record: Optional[Dict[str, Any]]
    problem: Optional[str]


def _shown_sequence(record: Any, index: int) -> str:
    sequence = _readable_sequence(record)
    return f"record {sequence}" if sequence is not None else f"the record at index {index}"


def find_run(
    ledger: Dict[str, Any],
    run_id: str,
    cohort_digest: Optional[str],
    checkpoint_digest: Optional[str],
) -> RunLookup:
    """**The exact same-run lookup** (policy rev 16, K8-6r.2): the record that
    names run `run_id` of cohort version `cohort_digest` for checkpoint
    `checkpoint_digest`, among the records with an explicit identity only, and
    before any other record is consulted. `append_record`, `held_record` and
    `report_state` all call it first, so the outcome, the record returned and
    the state agree.

    The records looked at: in a schema 4 file, those at or after its boundary,
    and a record whose sequence cannot be read that names a run; in a schema 4
    file whose boundary cannot be read, those that name a run readably; none in
    a schema 3 file. Among them:

    - a readable cohort digest that is not `cohort_digest`, or a readable
      checkpoint digest that is not `checkpoint_digest`, is not this run, and
      neither is a readable run id that is not `run_id`;
    - a record with no readable run id that may be this run (its digests are
      these or cannot be read): **problem**, naming its index;
    - a record naming this run whose cohort or checkpoint digest, whose
      report digest, or whose registration sequence cannot be read:
      **problem**, naming its index;
    - two records naming this run: **problem**, naming both;
    - with the boundary lost and no record naming this run: **problem**, saying
      the boundary cannot be read.

    `cohort_digest` or `checkpoint_digest` None means unknown (a report that
    could not be read): that filter is not applied. Never writes, never
    guesses: a problem is returned, not resolved.
    """
    boundary, why = run_identity_boundary(ledger)
    records = _ledger_records(ledger)
    problems: List[str] = []
    named: List[int] = []
    for index, (record, identity) in enumerate(zip(records, record_identities(ledger))):
        if not isinstance(record, dict) or identity.historical:
            continue
        if identity.run_id is None and (boundary is None or _readable_sequence(record) is None):
            # Not a record of this file's explicit identities as far as can be
            # told: the boundary is lost, or its sequence cannot be read (which
            # refuses an append, and leaves a state "cannot be checked", by
            # `sequence_problem`, once no record names this run).
            continue
        cohort = record.get("cohort_digest")
        checkpoint = record.get("checkpoint_digest")
        if ((cohort_digest is not None and _is_digest(cohort) and cohort != cohort_digest)
                or (checkpoint_digest is not None and _is_digest(checkpoint)
                    and checkpoint != checkpoint_digest)):
            continue
        if identity.run_id is not None and identity.run_id != run_id:
            continue
        if identity.run_id is None:
            problems.append(
                f"the ledger's record {index} should name its run and does not, so whether "
                "it is this run cannot be told")
            continue
        named.append(index)
        unreadable = [field for field, value in (("cohort digest", cohort),
                                                 ("checkpoint digest", checkpoint))
                      if not _is_digest(value)]
        if unreadable:
            problems.append(
                f"the ledger's record {index} names run {run_id} but its "
                f"{' and '.join(unreadable)} cannot be read, so whether it is this run's "
                "record cannot be told")
        elif not _is_digest(record.get("source_artifact_digest")):
            problems.append(
                f"{_shown_sequence(record, index)} (the ledger's record {index}) names run "
                f"{run_id} but its report digest cannot be read, so whether this report is "
                "the one registered cannot be told")
        elif _readable_sequence(record) is None:
            # This run's own record with its registration lost by a hand edit:
            # damaged identity, never an answer that it is registered as some
            # record (clarification 1).
            problems.append(
                f"the ledger's record {index} names run {run_id} but its registration "
                "sequence cannot be read, so this run's record is damaged; restore it from "
                "the original report or a trusted backup")
    if len(named) > 1:
        problems.append(
            f"the ledger's records {', '.join(str(i) for i in named)} all name run {run_id} "
            "(a hand edit, or a writer that took no lock), so which is this run's record "
            "cannot be told")
    if problems:
        return RunLookup(None, None, "; ".join(problems))
    if named:
        return RunLookup(named[0], records[named[0]], None)
    if boundary is None and why is not None:
        return RunLookup(None, None, (
            f"{_boundary_lost(why)}, so whether a record that names no run is this run "
            "cannot be told; restore the boundary from a trusted backup, then register "
            "again"))
    return RunLookup(None, None, None)


def sealed_run_refusal(found: RunLookup, run_id: str, source_digest: str) -> str:
    """Why run `run_id`'s report with bytes `source_digest` is not the one its
    record holds: the run's report is sealed when it is published, so other
    bytes are a replaced report. The writer's refusal and `report_state`'s
    reason, in one wording."""
    held = str(found.record.get("source_artifact_digest"))
    return (
        f"run {run_id} is already registered as {_shown_sequence(found.record, found.index)} "
        f"from report {held[:12]}...; this report ({source_digest[:12]}...) is other bytes: "
        "a run's report is sealed when it is published, so this file was replaced after "
        "it was registered"
    )


def append_record(
    ledger: Dict[str, Any], record: Dict[str, Any], *, run_id: str, registered_at: str
) -> Tuple[str, Dict[str, Any]]:
    """Register the record of run `run_id`, or recognise it, or refuse a write
    that cannot complete safely. Returns the outcome and the ledger to write.
    The steps, by the plan's numbers (K8-6r.2 (ii)(b)), in the order they run:

    1. **The exact same-run lookup** (`find_run`), before any other record is
       consulted: the same run with the same report bytes is
       **already_registered**, and the ledger is returned as it is; the same
       run with other bytes is refused (`sealed_run_refusal`); this run's own
       identity record damaged, or the boundary lost while no record names this
       run, is refused with its reason.
    0. **What makes the write itself uncertain** is refused before anything is
       appended, naming the record (`sequence_problem`): a record that is not
       an object or has no readable registration sequence, since the next
       sequence cannot be allocated safely otherwise. Checked once the lookup
       found no record of this run: a run it found is answered with nothing
       written, as `report_state` reads it, so a damaged record elsewhere never
       turns the same run's same report into a refusal.
    2. **Otherwise registered**: the record gets ``run_id``, no corroborating
       reports and ``registration`` {the largest sequence in the file plus one,
       from 1; ``registered_at``}, whatever its bytes and whatever the times. The
       ledger is returned as schema 4; when the file had no boundary (schema 3,
       no file, or an empty schema 4 file) its ``run_identity_from_sequence`` is
       this record's sequence. Every earlier record is left as written, with no
       run invented for it.

    **Nothing else is compared or consulted**: no metric, regime, environment,
    R10, cohort, relation or condition group, and no historical record. A lost
    group field, source digest or corroborating list on any other record never
    refuses. How this run relates to the historical records and to its group is
    described after the decision (`historical_overlap`, `group_differences`).

    Pure: the ledger and the record passed in are never changed.
    """
    if not isinstance(registered_at, str) or not registered_at:
        raise ValueError("a registration needs the time it was made")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            f"the run id {run_id!r} is not a run directory's name; a record names the run "
            "whose report it was built from"
        )
    for field in ("run_id", "corroborating_reports", "registration"):
        if field in record:
            raise ValueError(
                f"the record already carries {field}, which only the ledger assigns"
            )
    built = (*KEY_FIELDS, *RECORD_COMPARED_FIELDS, "r10", "environment",
             "source_artifact_digest")
    absent = [field for field in built if field not in record]
    if absent:
        raise ValueError(
            f"the record lacks {', '.join(absent)}; only a record build_record made "
            "is appended"
        )
    digest = record["source_artifact_digest"]
    records = ledger["records"]
    # 1. **This run first**, among the records that name a run: a run found is
    # answered here, and nothing is written for it either way.
    found = find_run(ledger, run_id, record["cohort_digest"], record["checkpoint_digest"])
    if found.record is not None:
        if found.record.get("source_artifact_digest") == digest:
            return "already_registered", ledger
        raise ValueError(sealed_run_refusal(found, run_id, digest))
    if found.problem is not None:
        raise ValueError(found.problem)
    # 0. **Every record's sequence reads before one is appended,** so a hand
    # edit that left one unreadable refuses the append wherever it sits.
    uncertain = sequence_problem(ledger)
    if uncertain is not None:
        raise ValueError(uncertain)
    # 2. **Its own record.**
    sequence = max(map(_readable_sequence, records), default=0) + 1
    registered = {
        **copy.deepcopy(record),
        "run_id": run_id,
        "corroborating_reports": [],
        "registration": {"sequence": sequence, "registered_at": registered_at},
    }
    updated = {**ledger, "schema_version": LEDGER_SCHEMA_VERSION,
               "records": records + [registered]}
    if not _schema_4(ledger) or RUN_IDENTITY_FROM_SEQUENCE not in ledger:
        updated[RUN_IDENTITY_FROM_SEQUENCE] = sequence
    return "registered", updated


def held_record(ledger: Dict[str, Any], record: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """The record `ledger` holds for run `run_id` of `record`'s cohort version
    and checkpoint (`find_run`): the one appended, or the one found for an
    ``already_registered`` run. Never the condition group's first record."""
    found = find_run(ledger, run_id, record.get("cohort_digest"),
                     record.get("checkpoint_digest"))
    if found.record is None:
        raise ValueError(found.problem or f"no record of this ledger names run {run_id}")
    return found.record


def _group(record: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """A record's condition group, or None when a group field is missing."""
    try:
        return record_key(record)
    except ValueError:
        return None


def _readable_group(record: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """A record's condition group when every field of it reads (the three
    digests as digests, the cohort role as a non-empty string); None when one
    is missing or cannot be read, so the group cannot be told."""
    group = _group(record)
    if group is None:
        return None
    readable = all(
        (isinstance(value, str) and bool(value)) if field == "cohort_role" else _is_digest(value)
        for field, value in zip(KEY_FIELDS, group))
    return group if readable else None


class HistoricalNote(NamedTuple):
    """One historical record's relation to a run's report (`historical_overlap`):
    its index and sequence (None when unreadable), whether it holds the report's
    digest (a possible overlap) or its relation cannot be checked, and the
    sentence for a person."""

    index: int
    sequence: Optional[int]
    possible_overlap: bool
    text: str


def historical_overlap(ledger: Dict[str, Any], record: Dict[str, Any]) -> Tuple[HistoricalNote, ...]:
    """**The historical note** (K8-6r.2 (ii)(b)3): how the records written before
    each run had its own record relate to `record`'s report. Informational,
    derived from the ledger alone, and never consulted to decide an outcome.

    - A historical record holding the report's digest among its readable source
      or corroborating digests: "possible overlap with record N, merged before
      each run had its own record: not recorded whether this run is one of them".
    - A historical record of `record`'s condition group, or of a group that
      cannot be read, whose source digest or corroborating list cannot be read:
      "its relation to record N cannot be checked: <field> cannot be read". One
      of another readable group gets no note: the old writer merged a report only
      into a record of its own group.

    Nothing is stored: historical records no longer change, so a stored flag
    would only repeat what the ledger says.
    """
    digest = record.get("source_artifact_digest")
    group = _group(record)
    notes: List[HistoricalNote] = []
    for index, (existing, identity) in enumerate(
            zip(_ledger_records(ledger), record_identities(ledger))):
        if not isinstance(existing, dict) or not identity.historical:
            continue
        sequence = _readable_sequence(existing)
        shown = _shown_sequence(existing, index)
        source = existing.get("source_artifact_digest")
        corroborating = existing.get("corroborating_reports")
        listed = corroborating if isinstance(corroborating, list) else []
        held = [d for d in [source, *listed] if _is_digest(d)]
        if _is_digest(digest) and digest in held:
            notes.append(HistoricalNote(index, sequence, True, (
                f"possible overlap with {shown}, merged before each run had its own record: "
                "not recorded whether this run is one of them")))
            continue
        # Another group only when every field of it reads: a group field that
        # is missing or unreadable leaves the group one that cannot be read.
        theirs = _readable_group(existing)
        if theirs is not None and theirs != group:
            continue
        lost = []
        if not _is_digest(source):
            lost.append("source_artifact_digest")
        if not isinstance(corroborating, list) or not all(_is_digest(d) for d in corroborating):
            lost.append("corroborating_reports")
        if lost:
            notes.append(HistoricalNote(index, sequence, False, (
                f"its relation to {shown} cannot be checked: {' and '.join(lost)} "
                "cannot be read")))
    return tuple(notes)


def possible_overlaps(ledger: Dict[str, Any]) -> List[Tuple[int, ...]]:
    """For each record, in file order, the registration sequences of the
    records it may overlap (K8-6r.2 (e)), for the readers that show records
    (the API's ``historical_overlap``, Models, ``record_evaluation --show``):

    - a record that names its run: each historical record holding its report
      digest among its readable source or corroborating digests;
    - a historical record: each record naming a run whose report digest it
      holds so.

    **Derived and informational**, by the same reading as `historical_overlap`
    (`record_identities`, the file's boundary): whether a historical record and
    a named run were one execution was never recorded, so an overlap is never
    confirmed, never counted as two independent runs and never a reason for
    anything. Empty for a record whose identity or digests cannot be read, and
    a record whose sequence cannot be read is not named."""
    records = _ledger_records(ledger)
    identities = record_identities(ledger)
    held_by: Dict[int, List[str]] = {}
    for index, (record, identity) in enumerate(zip(records, identities)):
        if isinstance(record, dict) and identity.historical:
            listed = record.get("corroborating_reports")
            listed = listed if isinstance(listed, list) else []
            held_by[index] = [d for d in [record.get("source_artifact_digest"), *listed]
                              if _is_digest(d)]
    out: List[List[int]] = [[] for _ in records]
    for index, (record, identity) in enumerate(zip(records, identities)):
        if not isinstance(record, dict) or identity.run_id is None:
            continue
        digest = record.get("source_artifact_digest")
        if not _is_digest(digest):
            continue
        for old, digests in held_by.items():
            if digest not in digests:
                continue
            old_sequence = _readable_sequence(records[old])
            new_sequence = _readable_sequence(record)
            if old_sequence is not None:
                out[index].append(old_sequence)
            if new_sequence is not None:
                out[old].append(new_sequence)
    return [tuple(sorted(set(found))) for found in out]


class GroupDifference(NamedTuple):
    """One other record of a run's condition group whose compared fields differ
    (`group_differences`): its sequence and index, its run (None: "not
    recorded" for a historical record, or unreadable), the difference names, and
    whether both runs record deterministic algorithms on (then they were
    expected to agree: investigate)."""

    index: int
    sequence: Optional[int]
    run_id: Optional[str]
    historical: bool
    differing: Tuple[str, ...]
    deterministic: bool


def _deterministic(record: Dict[str, Any]) -> bool:
    regime = record.get("regime")
    return isinstance(regime, dict) and regime.get("deterministic_algorithms") is True


def group_differences(ledger: Dict[str, Any], held: Dict[str, Any]) -> Tuple[GroupDifference, ...]:
    """Each other record of `held`'s condition group that differs from it on a
    compared field (`differing_fields`, named by `difference_names`), in file
    order. Informational (Decision 11): a description at registration, never a
    reason to refuse, and nothing is stored. `held` is the record the ledger
    holds for this run (`held_record`), found in `ledger` by identity."""
    group = _group(held)
    if group is None:
        return ()
    out: List[GroupDifference] = []
    for index, (existing, identity) in enumerate(
            zip(_ledger_records(ledger), record_identities(ledger))):
        if existing is held or not isinstance(existing, dict) or _group(existing) != group:
            continue
        fields = differing_fields(existing, held)
        if not fields:
            continue
        out.append(GroupDifference(
            index, _readable_sequence(existing), identity.run_id, identity.historical,
            tuple(difference_names(existing, held, fields)),
            _deterministic(existing) and _deterministic(held)))
    return tuple(out)


def differing_fields(existing: Dict[str, Any], record: Dict[str, Any]) -> List[str]:
    """The fields of ``RECORD_COMPARED_FIELDS`` on which two records of one
    condition group differ, in that list's order; empty when they agree.

    **The one comparison, and it only describes** (policy rev 16): the report
    store's description of a group (`StoreScan.status`) and the registration's
    group note (`group_differences`) both use it, so "runs under these
    conditions differ" means the same everywhere. It refuses nothing. A field
    one record lacks and the other has differs; a field both lack does not.
    """
    missing = object()
    return [
        field for field in RECORD_COMPARED_FIELDS
        if existing.get(field, missing) != record.get(field, missing)
    ]


def difference_names(
    existing: Dict[str, Any], record: Dict[str, Any], differing: List[str]
) -> List[str]:
    """How a difference is named for a person: the differing fields, or, when
    only the metrics differ, the metric names that do."""
    if differing != ["metrics"]:
        return list(differing)
    old = existing.get("metrics")
    old = old if isinstance(old, dict) else {}
    new = record.get("metrics")
    new = new if isinstance(new, dict) else {}
    return sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))


def report_key(report: Any) -> Optional[Tuple[Any, ...]]:
    """The key a report's record would be filed under, read from its manifest
    without `build_record`'s checks, or `None` when the manifest does not give
    all four parts.

    **For telling which group a report belongs to, never for filing it.** A
    report the store keeps that `build_record` refuses (written by an earlier
    revision, say) cannot be compared, but its group can often still be read, so
    the report store's description of that group says the comparison is
    incomplete, naming the file; it refuses nothing. For a report `build_record`
    accepts, this is the record's `record_key`, which a test pins.
    """
    manifest = report.get("manifest") if isinstance(report, dict) else None
    if not isinstance(manifest, dict):
        return None
    digests = manifest.get("artifact_digests")
    cohort_version = manifest.get("cohort_version")
    checkpoint = digests.get("checkpoint") if isinstance(digests, dict) else None
    cohort = cohort_version.get("manifest_sha256") if isinstance(cohort_version, dict) else None
    role = manifest.get("split")
    if not (_is_digest(checkpoint) and _is_digest(cohort) and isinstance(role, str)):
        return None
    # From a hand edit: a value JSON cannot hold, or one nested too deeply.
    try:
        semantics = measurement_semantics_digest(manifest)
    except (TypeError, ValueError, RecursionError):
        return None
    return (checkpoint, role, cohort, semantics)


def find_checkpoint(directory: Path, digest: str) -> Optional[Path]:
    """A file in this directory whose bytes are the ones the report measured.

    **Otherwise "beside the checkpoint" is not a fact.** A ledger written into a
    directory that does not hold the weights it describes is a record filed under
    the wrong address, and the reader's whole reason for looking there is gone.

    Candidates are hashed in sorted order and the search stops at the first match,
    so the full scan is paid only when there is no match — the case that ends in a
    refusal anyway. Several files with identical bytes are fine: the digest is the
    identity, not the name.

    A match may be a link: its bytes are its target's. Where the ledger then goes
    is the caller's to decide (`register_report` files it beside the target).
    """
    for candidate in sorted(directory.glob("*.pt")):
        if checkpoint_matches(candidate, digest):
            return candidate
    return None


def checkpoint_matches(path: Path, digest: str) -> bool:
    """Whether the file at ``path`` holds exactly the bytes ``digest`` names.

    The one comparison by which a later registration ties a report to weights on
    disk: `find_checkpoint` makes it for each candidate in a directory, and
    `register_report` for the one file ``record_evaluation.py --checkpoint``
    names, whatever its name. One hashing site, so the two cannot come to
    disagree about what "these weights" means.

    A path hash of a file nobody is about to parse (M2.1's allowance for
    `file_sha256`): the registration records the report's digest of the weights,
    never this one. A path that is not a regular file (missing, or a directory)
    does not match; a link matches by its target's bytes.
    """
    from src.utils.fingerprint import file_sha256

    return path.is_file() and file_sha256(path) == digest


def indexed_records_for(ledger: Dict[str, Any],
                        checkpoint_digest: str) -> List[Tuple[int, Dict[str, Any]]]:
    """`records_for`'s records, each with its index in the ledger's records
    list (a count from 0, the number the writer's refusals name a record by):
    the one predicate, so a reader that lists a record it cannot read names
    the record the file and the writer name."""
    records = ledger.get("records")
    if not isinstance(records, list):
        return []
    return [
        (index, record)
        for index, record in enumerate(records)
        if isinstance(record, dict) and record.get("checkpoint_digest") == checkpoint_digest
    ]


def records_for(ledger: Dict[str, Any], checkpoint_digest: str) -> List[Dict[str, Any]]:
    """Every record this ledger holds for one set of weights.

    An empty list is the honest answer to "what has this checkpoint been measured
    on", scoped to this ledger. It is never an answer to "has it been evaluated".

    **Tolerates hand edits.** Correcting a record means editing the file, so a
    record that lost its digest or is not an object is simply not one of this
    checkpoint's; it does not make the lookup raise. Whether a matching record is
    whole is the reader's to check before showing it (the readers' one listing,
    `src.api.routes.evaluations.read_records`, lists one that is not with its
    index and reason).
    """
    return [record for _index, record in indexed_records_for(ledger, checkpoint_digest)]


def environment_class(record_or_regime: Dict[str, Any]) -> str:
    """The class of regime a result was measured under, in one phrase.

    Takes a record or its ``regime`` block. CUDA runs are told apart by whether
    deterministic algorithms were on, because that says whether two runs of one
    group whose values differ were expected to agree (on: investigate) or may
    differ by variation the regime permits; neither is refused. A CPU run is a
    development number and a different measurement from any CUDA run (the device
    is in the key).

    A device that is neither, or a determinism flag that is not a boolean, is
    refused rather than classed: there is no class to guess. **The refusal
    names the field, never its value** (K8-6e.1): a reader lists the record
    with these words, and a hand-edited value can be a server path.
    """
    if not isinstance(record_or_regime, dict):
        raise ValueError("a record or a regime is an object")
    regime = record_or_regime.get("regime", record_or_regime)
    if not isinstance(regime, dict):
        raise ValueError("the record's regime is not an object")
    device = regime.get("device")
    kind = device.split(":", 1)[0] if isinstance(device, str) else None
    if kind == "cpu":
        return ENVIRONMENT_CLASSES[2]
    if kind != "cuda":
        raise ValueError("the regime's device is neither CUDA nor CPU")
    deterministic = regime.get("deterministic_algorithms")
    if type(deterministic) is not bool:
        raise ValueError("the regime's deterministic_algorithms is not a boolean")
    return ENVIRONMENT_CLASSES[0] if deterministic else ENVIRONMENT_CLASSES[1]


def ledger_digest(path: Path) -> Optional[str]:
    """The bytes a ledger holds now, or ``None`` if there is no file: the write
    guard's comparison, not a reader's identity (``read_ledger`` returns that)."""
    from src.utils.fingerprint import file_sha256

    return file_sha256(path)


def ledger_lock_path(path: Path) -> Path:
    """The lock file of the ledger at `path`: ``evaluations.json.lock`` beside
    it. A sibling file, because each write replaces the ledger with a new file
    (so a lock on the ledger itself would be on the old one), and an open
    ledger would block that replace on Windows."""
    path = Path(path)
    return path.with_name(path.name + LEDGER_LOCK_SUFFIX)


def ledger_lock(path: Path) -> AbstractContextManager[Path]:
    """Hold the lock of the ledger at `path` for a `with` block: **one
    registration at a time per ledger, across processes** (K8-6r.1).

    The registration holds it from reading the ledger to writing the new file
    and its usage-history entry (`registration.register_report`). It is keyed by
    the ledger's own path, beside the checkpoint, never by a setting, so
    processes started with different settings that write one ledger still take
    turns. It waits up to `LEDGER_LOCK_GRACE_SECONDS`, read now, and then raises
    `FileLockBusy`. The lock file is created when absent and never removed:
    removing it while another process holds it would let a third take a new
    file under the same name. Cooperative, on the local filesystem, untested on
    Windows (`src/utils/file_lock.py`).
    """
    return exclusive_lock(ledger_lock_path(path), grace=LEDGER_LOCK_GRACE_SECONDS)


def write_ledger(
    path: Path, ledger: Dict[str, Any], expected_digest: Optional[str]
) -> None:
    """Write the ledger, replacing the old file only once the new one is complete.

    **A partial write is closed.** Writing to a temporary file in the same
    directory and renaming it makes the replacement atomic on every platform
    this runs on, so an interruption cannot leave unparseable JSON where a
    directory's entire evaluation history used to be — records that are not
    recoverable from the checkpoints.

    **Concurrent writers take turns through `ledger_lock`, and this check is the
    second line** (K8-6r.1). The registration holds the ledger's lock from its
    read to this write, so another registration waits for it. ``expected_digest``
    is the digest ``read_ledger`` returned for the bytes it parsed, compared
    here with a hash of the file by path just before the replacement. It catches
    a writer that takes no lock (a hand edit, a build older than the lock) and
    replaced the file after this writer read it: that write is refused rather
    than erased. It is not compare-and-swap: such a writer that replaces the
    file between this comparison and the rename is still lost; a registration,
    which holds the lock, never is.

    The parameter is required rather than defaulted, so a caller cannot skip the
    check by forgetting it, and ``None`` carries a claim of its own: *I read no
    file*. A ledger appearing under a writer that expected none is another
    writer's creation, and is refused for the same reason.
    """
    if ledger_digest(path) != expected_digest:
        raise ValueError(
            f"{path} changed since it was read"
            + ("" if expected_digest is not None else " (this writer expected no "
               "file to exist there)")
            + ", so appending would erase whatever the other writer added. A "
            "registration holds the ledger's lock, so a writer that does not (a hand "
            "edit, an older build) changed it: re-read it and append again."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = create_staging_file(path, binary=False)
    try:
        with handle:
            json.dump(ledger, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


__all__ = [
    "COHORT_VERSION_FIELDS",
    "ENVIRONMENT_CLASSES",
    "GroupDifference",
    "HistoricalNote",
    "KEY_FIELDS",
    "LEDGER_FILENAME",
    "LEDGER_KINDS",
    "LEDGER_LOCK_GRACE_SECONDS",
    "LEDGER_LOCK_SUFFIX",
    "LEDGER_SCHEMA_VERSION",
    "MAX_REPORT_DEPTH",
    "NON_SEMANTIC_FIELDS",
    "READABLE_LEDGER_SCHEMAS",
    "RECORD_COMPARED_FIELDS",
    "RECORD_EXCLUDED_FIELDS",
    "REGIME_FIELDS",
    "REGISTERED_RELATIONS",
    "REGISTRATION_OUTCOMES",
    "RUN_IDENTITY_FROM_SEQUENCE",
    "RUN_ID_PATTERN",
    "RecordIdentity",
    "ReportNestsTooDeeply",
    "RunLookup",
    "SCORING_SEMANTICS_FIELDS",
    "SEMANTIC_ARTIFACT_ROLES",
    "SEMANTIC_MANIFEST_FIELDS",
    "append_record",
    "build_record",
    "checkpoint_matches",
    "difference_names",
    "differing_fields",
    "empty_ledger",
    "environment_class",
    "find_checkpoint",
    "find_run",
    "group_differences",
    "held_record",
    "historical_overlap",
    "indexed_records_for",
    "ledger_digest",
    "ledger_lock",
    "ledger_lock_path",
    "measurement_semantics_digest",
    "possible_overlaps",
    "read_ledger",
    "record_identities",
    "record_key",
    "records_for",
    "report_key",
    "run_identity_boundary",
    "scoring_semantics_digest",
    "sealed_run_refusal",
    "write_ledger",
]
