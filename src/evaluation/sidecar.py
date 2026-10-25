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

**What a record copies, and what it does not** (schema 3). A record copies what
the read-only view and D4's comparison need: the key, the scoring semantics and
the numerical regime, the cohort version's identity and aggregate counts, the
metrics, the relation states, R10 and the environment. The full report stays the
source: the record names it by digest, and the rest of the manifest, the rank
file and anything at case level stay there.

- **No run paths.** The manifest's `checkpoint_path` and `data_dir` stay in the
  report: they say where one run happened to find its files, and a checkpoint
  measured again from another place is the same measurement.
- **R10's evidence paths are kept.** The `r10` block is copied as the report
  has it (`src/evaluation/r10.py`), so each scope's `evidence.path` -- the file
  on the measuring machine whose bytes the checkpoint records, when one was
  found -- is in the record. The read-only view strips it
  (`src/api/routes/evaluations.py`); `record_evaluation.py --show` prints the
  record as it is, on the operator's machine.
- **No case ids.** The cohort block is aggregates only (`COHORT_VERSION_FIELDS`).
- **No prose.** A wording change between revisions would make an identical
  re-registration look like a contradiction, so the surfaces render the
  standing text from `src/evaluation/caveats.py` instead.

**Not a registry, and deliberately not becoming one.** No parent/child graph, no
lineage walk, no cross-file joins. The registration sequence orders the records
of this one file; nothing relates two ledgers.

**Contradiction is refused, not merged.** A key identifies one measurement. If
the same key arrives with different results, one of the two is wrong about what
it measured, and a ledger that silently kept the newer one would erase the
evidence that they disagreed. **The same result arriving again is not a
contradiction** (D5's duplicate rule): the same report is a no-op, and another
report whose compared fields all agree — a re-run, or a checkpoint measured again
after a move — adds its digest to the first record's corroborating reports. That
is a duplicate rule, not arbitration: nothing chooses between two results. There
is no override flag: correcting a bad record means editing a readable JSON file,
which leaves a diff, rather than passing a flag that leaves nothing.

Standard library only — no torch, no I/O beyond one file.

Module: src/evaluation/sidecar.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.file_modes import create_staging_file

#: Bumped when a record's shape changes in a way that makes an old file
#: unreadable under the new rules. A reader that finds a version it does not know
#: refuses rather than guessing which fields it can trust.
#:
#: **3 since M4.** A schema 3 record is keyed by the cohort version's manifest
#: digest rather than the samples digest, and carries what schema 2 lacked: the
#: relation states, the cohort's counts with their denominators, the scoring
#: semantics and regime, R10, the environment, a registration sequence and the
#: corroborating reports. A schema 2 record read under these rules would sit
#: beside them as though comparable with fields missing, so a schema 1 or 2 file
#: is refused and is read with the revision that wrote it. (2 was the Mode A/B
#: retirement's R2, which shortened the semantics field list.)
LEDGER_SCHEMA_VERSION = 3

#: The conventional file name, beside the checkpoints it describes.
LEDGER_FILENAME = "evaluations.json"

#: Manifest fields that can change a number, hashed into one semantics digest.
#:
#: **A false contradiction is worse than a missed one, and this list is chosen on
#: that asymmetry.** A missed contradiction leaves two records the reader can
#: compare; a false one *blocks a legitimate append*, so the rule is to include
#: anything that could move a metric. An earlier key of
#: (checkpoint, cohort, mode, tie policy) failed exactly this way: two honest runs
#: at different batch sizes collided and the second was refused as a
#: contradiction, although `batch_size` is the chunk Mode C scores at once and can
#: move its numbers at floating-point level.
#:
#: Explicit rather than "every field", so that adding a non-semantic field later
#: does not churn every key and silence the contradiction check. That claim is
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

#: D4's scoring semantics: what makes two results rankings of the same kind, so a
#: comparison across checkpoints groups by these and nothing narrower. A record
#: carries them and their own digest (`scoring_semantics_digest`).
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
#: records in one scoring group can differ here, and D4 shows that difference
#: rather than hiding it. With ``SCORING_SEMANTICS_FIELDS`` it partitions
#: ``SEMANTIC_MANIFEST_FIELDS``, which a test checks.
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
        "and PyG facts inform an investigation of two disagreeing runs (D5)"
    ),
}

#: Artifact roles whose bytes change what was measured. The graph tensors and the
#: allocation are as much a part of the measurement as the checkpoint is.
SEMANTIC_ARTIFACT_ROLES: Tuple[str, ...] = (
    "node_features", "edge_indices", "num_nodes", "split_manifest",
)

#: What makes two records the same measurement.
#:
#: ``mode`` was in an earlier version of this key and is now inside the semantics
#: digest with everything else that can move a number. What remains outside it are
#: the two identities a reader looks a record up by — whose weights, and which
#: cohort — and the cohort's role. Since schema 3 the cohort is the **cohort
#: version's manifest digest**: it binds the samples, the source, the graph, the
#: mapping policy and the counts, so two versions with identical samples but
#: different sources are two cohorts (D4: one cohort version per comparison).
KEY_FIELDS: Tuple[str, ...] = (
    "checkpoint_digest",
    "cohort_role",
    "cohort_digest",
    "measurement_semantics_digest",
)

#: The relation states a registered result carries: R1, R3, R5 and the cohort's
#: binding to the graph verified, and R2, a generated split's relation, not
#: applicable. Restated from ``expected_relations(registered=True)`` in
#: `src/evaluation/measurement.py` because this module imports no torch;
#: `tests/unit/test_evaluation_sidecar.py` fails if the two differ.
REGISTERED_RELATIONS: Dict[str, str] = {
    "R1": "verified",
    "R2": "not_applicable",
    "R3": "verified",
    "R5": "verified",
    "cohort_graph": "verified",
}

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

#: Record fields that must agree for a second report under one key to corroborate
#: the first rather than contradict it (D5's duplicate rule).
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
#: registered record, which a test checks, so a field cannot escape the duplicate
#: rule by oversight.
RECORD_EXCLUDED_FIELDS: Dict[str, str] = {
    "r10": (
        "a re-run may find evidence the first did not, so two runs of one "
        "measurement can carry different labels; each record's labels are true of "
        "its own report"
    ),
    "environment": (
        "R11: recorded, never refused. A difference informs an investigation of "
        "two runs; it does not make the second a contradiction"
    ),
    "source_artifact_digest": (
        "the report's bytes, which can differ between two runs of one measurement "
        "(their paths, their environment) and can also be identical (the same run "
        "repeated on one machine); it is what the duplicate rule adds to the "
        "corroborating reports"
    ),
    "corroborating_reports": "the ledger's own record of later agreeing reports",
    "registration": "when and in what order this ledger registered the record",
}

#: The excluded fields each report measures for itself. When a report
#: corroborates a record, the record keeps its first report's values of these,
#: so a re-run that found R10 evidence the first did not, or ran under another
#: driver, changes nothing in the ledger. A registration says so when this
#: report's values differ (`RegistrationResult.excluded_differences`), naming
#: the report that holds them, rather than reporting plain agreement.
CORROBORATION_KEEPS_FIRST: Tuple[str, ...] = ("r10", "environment")

#: What ``append_record`` can do with a record, by name.
REGISTRATION_OUTCOMES: Tuple[str, ...] = ("registered", "already_registered", "corroborated")

#: How a record's regime is described where a person reads it: the device class
#: and whether deterministic algorithms were on, which is what decides whether two
#: disagreeing runs are a contradiction or variation to investigate.
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
    reordered dict or an added descriptive field, so using it as identity would
    make every re-run a new record and the contradiction check would never fire.
    This hashes a named list of semantic fields, so two runs that differ in
    nothing that matters produce the same digest.

    **What "the same measurement" means here is narrower than bit-determinism.**
    Two records under one digest were produced under identical recorded
    semantics. Whether they were therefore *bound* to agree
    depends on the regime those fields describe: under `deterministic_algorithms`
    a disagreement is a contradiction, and under a non-deterministic CUDA regime
    it is a finding to investigate. The ledger refuses the second record either
    way, because two answers filed under one identity cannot both stand — but the
    refusal is an instruction to look, not a proof of a defect.

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
    still the same scoring, so D4 groups them and shows the regime difference.
    Absent fields hash as ``null``, as there.
    """
    return _canonical_digest(
        {field: manifest.get(field, None) for field in SCORING_SEMANTICS_FIELDS}
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


def read_ledger(path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load a ledger and the digest of the bytes parsed, or an empty one and
    ``None`` where no file exists.

    **One read** (BACKLOG item 18): the ledger is parsed from the bytes whose
    digest is returned, so a writer's stale-write guard compares the file with
    what it actually parsed. Reading the file and hashing it again let another
    writer's append land in between and be erased by a write that passed the
    guard.

    A missing file is not an error: the first record has to be appendable to a
    directory that has none. A file that is not UTF-8 JSON, not an object, has
    no records list or carries another ``schema_version`` **is** an error, named
    by its path, because reading it under this version's rules would be a guess
    about which of its fields still mean what they say.
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
    if type(version) is int and version in (1, 2):
        # **Not "register again".** No report written before schema 3 can be
        # registered under it (`build_record` refuses its manifest), and this text
        # is shown where no current report is at hand (the model status,
        # `--show`), so it says what can be done: re-measure.
        raise ValueError(
            f"{path} is ledger schema {version}; this code reads schema "
            f"{LEDGER_SCHEMA_VERSION}. Archive it, for example as "
            f"evaluations.v{version}.json; it is read with the revision that wrote "
            "it. Its results are not carried over: re-measure with this revision "
            "and register the new reports."
        )
    if type(version) is not int or version != LEDGER_SCHEMA_VERSION:
        raise ValueError(
            f"{path} is ledger schema {version!r}; this code reads "
            f"{LEDGER_SCHEMA_VERSION}. Read it with the revision that wrote it "
            "rather than reinterpreting its fields under new rules."
        )
    if not isinstance(ledger.get("records"), list):
        raise ValueError(f"{path} has no records list")
    return ledger, digest


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
    is refused: the ledger records results on test cohort versions (D3 labels a
    result by role *and version*), and a generated `val` is selection data. A
    report whose relation states are not a registered result's is refused too;
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
            "records results on registered test cohort versions, and a generated "
            "split is selection data. Measure a registered version "
            "(--cohort LABEL --registry DIR) to record a result."
        )
    if manifest["cohort_kind"] != "supplied":
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


def _sequence(record: Dict[str, Any], index: int) -> int:
    """A record's registration sequence, refused rather than guessed when a hand
    edit left it unreadable: a new sequence computed past a gap could repeat one."""
    registration = record.get("registration")
    sequence = registration.get("sequence") if isinstance(registration, dict) else None
    if type(sequence) is not int or sequence < 1:
        raise ValueError(
            f"the ledger's record {index} has no registration sequence; it was edited "
            "into a shape this code cannot append to without guessing"
        )
    return sequence


def append_record(
    ledger: Dict[str, Any], record: Dict[str, Any], *, registered_at: str
) -> Tuple[str, Dict[str, Any]]:
    """Register a record, or recognise it, or refuse a key that already carries a
    different result. Returns the outcome and the ledger to write.

    - **registered**: no record has this key. It is appended with the next
      sequence (the largest in the file plus one, from 1), no corroborating
      reports, and ``registered_at``.
    - **already_registered**: a record under this key was built from the same
      report bytes, or lists them among its corroborating reports. The ledger is
      returned as it is: a ledger rebuilt from the same reports twice is the same
      ledger.
    - **corroborated**: a record under this key came from another report, and
      every field in ``RECORD_COMPARED_FIELDS`` agrees. The new report's digest
      is added to that first record's corroborating reports; the first record
      keeps its R10, environment and registration.

    Anything else under the key is refused, naming the compared fields that
    differ (the metric names, when only the metrics do) — not because one of the
    two is provably wrong, but because the ledger cannot choose between them and
    neither may stand unexamined. Whether the difference is a defect or
    variation the regime permits is a question for a person:
    `deterministic_algorithms` and the cuDNN fields inform that investigation
    rather than settling it.

    Pure: the ledger and the record passed in are never changed.
    """
    if not isinstance(registered_at, str) or not registered_at:
        raise ValueError("a registration needs the time it was made")
    for field in ("corroborating_reports", "registration"):
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
    key = record_key(record)
    digest = record["source_artifact_digest"]
    records = ledger["records"]
    # **Every record is checked before any is compared,** so a hand edit that left
    # one unreadable refuses the append wherever it sits in the file.
    sequences = []
    for index, existing in enumerate(records):
        if not isinstance(existing, dict):
            raise ValueError(f"the ledger's record {index} is not an object")
        try:
            record_key(existing)
        except ValueError as exc:
            raise ValueError(f"the ledger's record {index}: {exc}") from exc
        sequences.append(_sequence(existing, index))
    for index, existing in enumerate(records):
        if record_key(existing) != key:
            continue
        corroborating = existing.get("corroborating_reports")
        if not isinstance(corroborating, list):
            raise ValueError(
                f"the ledger's record {index} has no corroborating_reports list"
            )
        if digest == existing.get("source_artifact_digest") or digest in corroborating:
            return "already_registered", ledger
        differing = differing_fields(existing, record)
        if not differing:
            updated = {**existing, "corroborating_reports": corroborating + [digest]}
            return "corroborated", {
                **ledger, "records": records[:index] + [updated] + records[index + 1:],
            }
        raise ValueError(_contradiction(existing, record, differing))
    registered = {
        **copy.deepcopy(record),
        "corroborating_reports": [],
        "registration": {
            "sequence": max(sequences, default=0) + 1,
            "registered_at": registered_at,
        },
    }
    return "registered", {**ledger, "records": records + [registered]}


def differing_fields(existing: Dict[str, Any], record: Dict[str, Any]) -> List[str]:
    """The fields of ``RECORD_COMPARED_FIELDS`` on which two records under one key
    disagree, in that list's order; empty when one corroborates the other.

    **The ledger's own comparison, and the only one** (D5's duplicate rule):
    `append_record` uses it against the record a ledger holds, and the report
    store uses it between the reports it keeps under one key
    (`src/evaluation/report_store.py`), so "runs disagree" means in the store
    exactly what a contradiction means in the ledger. A field one record lacks
    and the other has differs; a field both lack does not.
    """
    missing = object()
    return [
        field for field in RECORD_COMPARED_FIELDS
        if existing.get(field, missing) != record.get(field, missing)
    ]


def difference_names(
    existing: Dict[str, Any], record: Dict[str, Any], differing: List[str]
) -> List[str]:
    """How a disagreement is named for a person: the differing fields, or, when
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

    **For telling which key a report belongs to, never for filing it.** A report
    the store keeps that `build_record` refuses (written by an earlier revision,
    say) cannot be compared, but its key can often still be read, so the report
    store marks only that key "cannot be checked" rather than every key of the
    version (D5). For a report `build_record` accepts, this is the record's
    `record_key`, which a test pins.
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


def _contradiction(
    existing: Dict[str, Any], record: Dict[str, Any], differing: List[str]
) -> str:
    """The refusal of a second, different result under one key."""
    names = difference_names(existing, record, differing)
    return (
        f"this ledger already holds a record for checkpoint "
        f"{record['checkpoint_digest'][:12]}... on cohort version "
        f"{record['cohort']['label']} ({record['cohort_digest'][:12]}...) under "
        f"identical measurement semantics "
        f"({record['measurement_semantics_digest'][:12]}...), from report "
        f"{str(existing.get('source_artifact_digest'))[:12]}..., and this report "
        f"({record['source_artifact_digest'][:12]}...) disagrees on: "
        f"{', '.join(names)}. The ledger cannot choose between them, and neither "
        "may stand unexamined: whether this is a defect or variation the recorded "
        "regime permits is for a person to determine."
    )


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


def records_for(ledger: Dict[str, Any], checkpoint_digest: str) -> List[Dict[str, Any]]:
    """Every result this ledger holds for one set of weights.

    An empty list is the honest answer to "what has this checkpoint been measured
    on", scoped to this ledger. It is never an answer to "has it been evaluated".

    **Tolerates hand edits.** Correcting a record means editing the file, so a
    record that lost its digest or is not an object is simply not one of this
    checkpoint's; it does not make the lookup raise. Whether a matching record is
    whole is the reader's to check before showing it.
    """
    records = ledger.get("records")
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("checkpoint_digest") == checkpoint_digest
    ]


def environment_class(record_or_regime: Dict[str, Any]) -> str:
    """The class of regime a result was measured under, in one phrase.

    Takes a record or its ``regime`` block. CUDA runs are told apart by whether
    deterministic algorithms were on, because that decides whether two
    disagreeing runs under one key are a contradiction or variation to
    investigate (D5). A CPU run is a development number and a different
    measurement from any CUDA run (the device is in the key).

    A device that is neither, or a determinism flag that is not a boolean, is
    refused rather than classed: there is no class to guess.
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
        raise ValueError(f"the regime's device is {device!r}, neither CUDA nor CPU")
    deterministic = regime.get("deterministic_algorithms")
    if type(deterministic) is not bool:
        raise ValueError(
            f"the regime's deterministic_algorithms is {deterministic!r}, not a boolean"
        )
    return ENVIRONMENT_CLASSES[0] if deterministic else ENVIRONMENT_CLASSES[1]


def ledger_digest(path: Path) -> Optional[str]:
    """The bytes a ledger holds now, or ``None`` if there is no file: the write
    guard's comparison, not a reader's identity (``read_ledger`` returns that)."""
    from src.utils.fingerprint import file_sha256

    return file_sha256(path)


def write_ledger(
    path: Path, ledger: Dict[str, Any], expected_digest: Optional[str]
) -> None:
    """Write the ledger, replacing the old file only once the new one is complete.

    **A partial write is closed; concurrent append is not, and that is stated
    rather than implied.** Writing to a temporary file in the same directory and
    renaming it makes the replacement atomic on every platform this runs on, so
    an interruption cannot leave unparseable JSON where a directory's entire
    evaluation history used to be — records that are not recoverable from the
    checkpoints.

    ``expected_digest`` is **best-effort stale-write detection, not mutual
    exclusion.** It is the digest ``read_ledger`` returned for the bytes it
    parsed, compared here with a hash of the file by path just before the
    replacement. It catches the ordinary accident: a writer that read the ledger,
    another writer replaced it, and the first then writes over that. It does
    **not** make concurrent append safe — two writers whose check-then-replace
    windows overlap both pass the comparison and the second replacement still
    erases the first append, with no trace. It is not compare-and-swap. Closing
    that needs real exclusion, which is not built: the ledger is
    **single-writer by operational rule**, and locking earns its place only if
    the institutional workflow actually appends concurrently.

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
            + ", so appending would erase whatever the other writer added. The "
            "ledger is single-writer: re-read it and append again."
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
    "CORROBORATION_KEEPS_FIRST",
    "ENVIRONMENT_CLASSES",
    "KEY_FIELDS",
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA_VERSION",
    "MAX_REPORT_DEPTH",
    "NON_SEMANTIC_FIELDS",
    "RECORD_COMPARED_FIELDS",
    "RECORD_EXCLUDED_FIELDS",
    "REGIME_FIELDS",
    "REGISTERED_RELATIONS",
    "REGISTRATION_OUTCOMES",
    "ReportNestsTooDeeply",
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
    "ledger_digest",
    "measurement_semantics_digest",
    "read_ledger",
    "record_key",
    "records_for",
    "report_key",
    "scoring_semantics_digest",
    "write_ledger",
]
