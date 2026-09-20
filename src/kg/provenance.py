"""What a knowledge graph was built from, recorded beside the graph itself.

**A digest says which bytes; it does not say what produced them.** `kg.json` has
had a digest since the manifest bound it, and nothing anywhere said which MONDO
release, which HPO release or which annotation files went into it. Two sites
whose ontologies differ by a deployment date detect the divergence — their
`kg_digest` values differ — and cannot say what diverged, which is detection
without diagnosis.

**The record names its graph.** It carries the SHA-256 of the `kg.json` written
in the same build, so a record that has been separated from its graph and placed
beside another can be told apart from one that belongs. Without that, two
graph-only workspaces whose provenance files were swapped during a copy would
each read as describing the other's inputs, with every file well-formed.

**What it does not claim.** This is a record of the *source files* a build
consumed. It is not a rebuild recipe: the parser version, the builder's
parameters, and any ontology imports resolved at parse time are outside it, and
§`incomplete_by_design` says so in the file rather than leaving a reader to
assume otherwise.

Module: src/kg/provenance.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

#: The file, beside `kg.json`, that records what the graph was built from.
PROVENANCE_FILENAME = "kg.provenance.json"

#: Bumped when the record's shape changes in a way that makes an old file
#: unreadable under the new rules. A reader that does not recognise a version
#: reports that, rather than guessing which fields still mean what they say.
PROVENANCE_SCHEMA_VERSION = 1

#: The roles a real build consumes. Declared so a record missing one is visible
#: as incomplete rather than merely short, and so a fifth input is a deliberate
#: schema change instead of a silent omission.
SOURCE_ROLES = ("mondo", "hpo", "phenotype_hpoa", "genes_to_phenotype")

#: What this record deliberately does not establish. Written into every file so
#: the limits travel with it and a later reader cannot mistake a source list for
#: a reproduction recipe.
INCOMPLETE_BY_DESIGN = (
    "ontology imports resolved at parse time are not identified",
    "parser and builder versions are not recorded",
    "graph construction parameters are not recorded",
)


#: The fields a usable record carries. "Parses as JSON" is not "is a provenance
#: record": one with only a schema version and a digest states no origin and
#: lists no sources, and a reader that accepted it would report a build's inputs
#: as an empty set rather than as unrecorded.
REQUIRED_RECORD_FIELDS = ("schema_version", "kg_digest", "origin", "sources")


class ProvenanceError(ValueError):
    """A provenance record that cannot be trusted to describe its graph."""


class ProvenanceStatus(NamedTuple):
    """What is known about a workspace's record, without deciding anything.

    **Separate from the graph and tensor bindings on purpose.** Those gate
    whether a workspace can be consumed at all and raise; this one answers a
    different question — what this graph was built from — and a caller that
    cannot serve without an answer is making a policy choice no approved plan
    has taken. Returning a state rather than raising is what keeps that choice
    with the caller.
    """

    #: absent | recorded | undeclared_present | missing | unreadable | mismatched
    state: str
    record: Optional[Dict[str, Any]]
    detail: str

    @property
    def is_known(self) -> bool:
        """True only when a record was found and holds for this graph."""
        return self.state == "recorded"


def source_entry(
    role: str,
    path: Any,
    digest: str,
    declared_version: Optional[str] = None,
) -> Dict[str, Any]:
    """One consumed file, as it goes into the record.

    `path` is kept as a **locator, not an identity** — a directory name can be
    reused across vintages and a file replaced in place under it, which is
    exactly how two builds diverge while looking alike. The digest is what
    identifies the bytes. Only the file's basename is kept, because the absolute
    path of an operator's home directory is not evidence and §5.2 of the backlog
    keeps it out of artifacts.

    `declared_version` is the raw `data-version` an ontology header carries, or
    None. **It is never filled in from a format version**: `Ontology.version`
    falls back to `format_version` and then to `"Unknown"`, so reading that
    property would let a file's OBO format masquerade as a release.
    """
    if role not in SOURCE_ROLES:
        raise ProvenanceError(
            f"{role!r} is not one of the roles a build consumes ({', '.join(SOURCE_ROLES)})"
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise ProvenanceError(
            f"the digest recorded for {role} is not a SHA-256 hexdigest ({digest!r})"
        )
    if declared_version is not None and not isinstance(declared_version, str):
        raise ProvenanceError(
            f"{role} declares a version that is not a string ({declared_version!r})"
        )
    return {
        "role": role,
        "filename": Path(path).name if path is not None else None,
        "digest": digest,
        "declared_version": declared_version,
    }


def build_provenance(
    kg_digest: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    counters: Optional[Dict[str, Any]] = None,
    origin: str = "files",
) -> Dict[str, Any]:
    """Assemble the record for one build.

    Args:
        kg_digest: SHA-256 of the `kg.json` this build wrote. The binding that
            makes the record about *this* graph rather than a plausible one.
        sources: `source_entry` results for the files actually consumed.
        counters: parsing statistics, each named by what it actually counts.
        origin: `"files"` when real inputs produced the graph, `"synthetic"`
            when it was constructed in memory by a demo or a test. **A synthetic
            build records that it has no sources; it never invents digests.**
    """
    if origin not in ("files", "synthetic"):
        raise ProvenanceError(f"origin must be 'files' or 'synthetic', got {origin!r}")
    if not isinstance(kg_digest, str) or len(kg_digest) != 64:
        raise ProvenanceError(
            f"kg_digest is not a SHA-256 hexdigest ({kg_digest!r}); without it "
            "the record cannot say which graph it describes"
        )
    entries = list(sources or [])
    # **Entries are re-checked here, not trusted because `source_entry` exists.**
    # That helper validates what it builds; nothing stops a caller assembling a
    # dict by hand, and a record listing an entry with no digest identifies
    # nothing while looking like it does.
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProvenanceError(f"a source entry is not an object ({entry!r})")
        absent = [key for key in ("role", "digest") if key not in entry]
        if absent:
            raise ProvenanceError(
                f"a source entry is missing {absent}; without them it names "
                f"nothing and identifies nothing ({entry!r})"
            )
        source_entry(
            role=entry["role"],
            path=entry.get("filename"),
            digest=entry["digest"],
            declared_version=entry.get("declared_version"),
        )
    if origin == "synthetic" and entries:
        raise ProvenanceError(
            "a synthetic build recorded source files. Nothing read them, so "
            "recording them would attribute a graph to inputs it never had."
        )
    missing = [role for role in SOURCE_ROLES if role not in {e["role"] for e in entries}]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kg_digest": kg_digest,
        "origin": origin,
        "sources": entries,
        # Named rather than implied: a reader can tell "this build had no real
        # inputs" from "this build had them and one went unrecorded".
        "missing_roles": missing if origin == "files" else list(SOURCE_ROLES),
        "counters": dict(counters or {}),
        "incomplete_by_design": list(INCOMPLETE_BY_DESIGN),
    }


def encode_provenance(record: Dict[str, Any]) -> str:
    """The record as the bytes that go to disk. **The one encoder.**

    A pre-write gate that asks "does this encode?" with different arguments
    from the writer is not a gate, it is a second encoder that happens to agree
    on most inputs. Measured on the pair this replaces: a counter keyed
    `{1: 2, "OMIM": 3}` passed the gate's `json.dumps(record)` and then failed
    the writer's `sort_keys=True` with `TypeError: '<' not supported between
    instances of 'str' and 'int'` — after `kg.json` and three tensors had been
    written. Both sides call this, so both ask the same question.

    Raises:
        TypeError, ValueError: whatever the encoder raises, unchanged. The
            caller that has not written anything yet turns it into a refusal;
            the caller mid-write lets it abort. Neither is decided here.
    """
    return json.dumps(record, indent=2, sort_keys=True)


def write_provenance(workspace: Path, record: Dict[str, Any]) -> str:
    """Serialise the record whole, then rename it into place.

    Returns its SHA-256, so the caller can bind it into a manifest without
    re-reading the file. Same whole-then-replace shape as the split manifest, and
    for the same reason: a value no encoder takes must not leave a truncated file
    that parses as nothing and reads as a record that exists.
    """
    from src.utils.fingerprint import file_sha256

    payload = encode_provenance(record)
    target = Path(workspace) / PROVENANCE_FILENAME
    handle = tempfile.NamedTemporaryFile(
        "w", dir=str(workspace), prefix=target.name, suffix=".tmp",
        delete=False, encoding="utf-8",
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return file_sha256(target)


def read_provenance(workspace: Path) -> Optional[Dict[str, Any]]:
    """The file, read and checked for shape. **Low-level; reports no state.**

    None here means only *no file is present in this directory*. It is not the
    answer to "was a record ever declared" — a manifest can declare one that has
    since been deleted, and this function does not read manifests, so it cannot
    tell that workspace from one built before provenance existed. Callers that
    need that distinction use `provenance_status`, which takes the declaration
    as an input; a caller that reads this `None` as "never declared" will give
    two opposite situations the same name.

    Raises:
        ProvenanceError: the file is there and unusable — unopenable, undecodable,
            unparseable, or parsed and not a record. Each is a different state
            from its absence and none of them is folded into one.
    """
    path = Path(workspace) / PROVENANCE_FILENAME
    if not path.is_file():
        return None
    # **Opening and parsing are separated because they fail for unlike reasons.**
    # A file that cannot be opened at all -- permissions, a directory in its
    # place, a device error -- is not malformed JSON, and a message saying it is
    # sends a reader to edit a file they cannot even read. Both are raised as
    # `ProvenanceError` so a status caller catches one type; the message says
    # which happened.
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProvenanceError(
            f"{path} is present and could not be read as UTF-8 text "
            f"({type(exc).__name__}: {exc}); a record that cannot be opened is "
            "not an absent one"
        ) from exc
    try:
        record = json.loads(text)
    except ValueError as exc:
        raise ProvenanceError(
            f"{path} is present but not readable JSON ({type(exc).__name__}); "
            "a record that cannot be parsed is not an absent one"
        ) from exc
    if not isinstance(record, dict):
        raise ProvenanceError(f"{path} is not a JSON object, so it describes no build")
    version = record.get("schema_version")
    if version != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError(
            f"{path} is provenance schema {version!r}; this code reads "
            f"{PROVENANCE_SCHEMA_VERSION}. Which of its fields still mean what "
            "they say is a guess."
        )
    absent = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if absent:
        raise ProvenanceError(
            f"{path} is missing {absent}. A file that parses is not a record: "
            "without them it states no origin and lists no sources, which reads "
            "as a build with no inputs rather than as inputs unrecorded."
        )
    return record


def provenance_status(
    workspace: Path,
    kg_digest: str,
    declared_digest: Optional[str] = None,
) -> ProvenanceStatus:
    """What is known about this workspace's record. **Never raises.**

    **Including when the disk refuses.** Every read this performs is an
    operation that can fail for reasons unrelated to what is recorded: a file
    mode that excludes this process, a truncated volume, a directory where a
    file belongs. Those come back as `unreadable` carrying the cause, because a
    caller told this never raises will not have wrapped the call, and a
    `PermissionError` escaping a reporting function stops a pipeline over a note
    that was never allowed to stop one.

    **The declaration is an input, because absence alone cannot be read.** A
    missing file means "this build predates provenance" only when nothing said
    there should be one. When a manifest declares the record's digest and the
    file is gone, that is a broken workspace, and a reader that reported both as
    `unknown` would give two names to opposite situations. `declared_digest` is
    how the caller supplies what it knows; `None` means nothing declared one.

    Args:
        workspace: the directory to inspect.
        kg_digest: the digest of the `kg.json` actually present.
        declared_digest: the digest a manifest recorded, when one did.

    Returns:
        A `ProvenanceStatus`. Deciding what to do about a state that is not
        `recorded` belongs to the caller: this establishes what is true, not
        whether to serve.
    """
    from src.utils.fingerprint import file_sha256

    path = Path(workspace) / PROVENANCE_FILENAME
    if not path.is_file():
        if declared_digest is None:
            return ProvenanceStatus(
                "absent", None,
                "no record, and nothing declared one: this workspace predates "
                "provenance and its inputs are unrecorded",
            )
        return ProvenanceStatus(
            "missing", None,
            f"{PROVENANCE_FILENAME} was declared and is not here; what this "
            "graph was built from was recorded once and has been lost",
        )

    if declared_digest is not None:
        try:
            observed = file_sha256(path)
        except OSError as exc:
            return ProvenanceStatus(
                "unreadable", None,
                f"{PROVENANCE_FILENAME} is here and could not be read "
                f"({type(exc).__name__}: {exc}), so whether it is the record "
                "that was declared cannot be established",
            )
        if observed != declared_digest:
            return ProvenanceStatus(
                "mismatched", None,
                f"{PROVENANCE_FILENAME} is not the record that was declared "
                f"({str(declared_digest)[:12]}... vs {observed[:12]}...); it was "
                "replaced after the build",
            )

    try:
        record = read_provenance(workspace)
    except ProvenanceError as exc:
        return ProvenanceStatus("unreadable", None, str(exc))
    if record is None:  # pragma: no cover - the file existed a moment ago
        return ProvenanceStatus("absent", None, "the record disappeared while being read")

    if record.get("kg_digest") != kg_digest:
        return ProvenanceStatus(
            "mismatched", None,
            f"{PROVENANCE_FILENAME} describes a graph whose digest is "
            f"{str(record.get('kg_digest'))[:12]}..., and the kg.json beside it "
            f"digests to {kg_digest[:12]}...; this record belongs to another build",
        )

    if declared_digest is None:
        return ProvenanceStatus(
            "undeclared_present", record,
            "a record is here and matches this graph, but nothing binds it — "
            "no manifest declared it, so it could have been placed here",
        )
    return ProvenanceStatus("recorded", record, "the record belongs to this graph")


def verify_provenance(workspace: Path, kg_digest: str) -> Optional[Dict[str, Any]]:
    """The record, checked against the graph actually present.

    **This is the whole point of the file.** A record names the `kg.json` it was
    written beside; a reader that trusts the directory instead would accept one
    workspace's inputs as another's, which is the failure a copy produces and
    every file involved looks correct.

    Returns the record when it belongs to this graph, or None when **no file is
    present in this directory** — which is not the same as "no record was ever
    declared", a question only a manifest can answer and this function does not
    read one. Raises when a file is present and does not hold — **a mismatch is
    reported as a mismatch and never folded into "unknown"**, because the two
    states mean opposite things about what is known.

    Verifying provenance says nothing about whether a pipeline should serve.
    Inference does not consume this record, and what a caller does about a
    broken claim is that caller's decision.
    """
    record = read_provenance(workspace)
    if record is None:
        return None
    recorded = record.get("kg_digest")
    if recorded != kg_digest:
        raise ProvenanceError(
            f"{Path(workspace) / PROVENANCE_FILENAME} records a graph whose "
            f"digest is {str(recorded)[:12]}..., and the kg.json beside it "
            f"digests to {kg_digest[:12]}.... This record describes another "
            "build; reading its sources as this graph's would attribute inputs "
            "it never had."
        )
    return record


__all__ = [
    "INCOMPLETE_BY_DESIGN",
    "PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceError",
    "ProvenanceStatus",
    "REQUIRED_RECORD_FIELDS",
    "SOURCE_ROLES",
    "build_provenance",
    "encode_provenance",
    "provenance_status",
    "read_provenance",
    "source_entry",
    "verify_provenance",
    "write_provenance",
]
