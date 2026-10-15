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
assume otherwise. That list is displayed beside the record and never refused;
closing it is the KG build record v2.

**Required since contract M3b.** Training, serving and measurement refuse a
workspace whose record is missing, unreadable or does not match: the manifest
binds the record's digest, and the record names the graph's
(`src.kg.artifacts.verify_dataset_record`, which reads it once). This module
holds the record's own rules, with no file around them, so the writer and that
reader apply the same ones (`validate_record`, `unidentified_roles`).

**Two facts about a record, kept apart.** Its *binding state* — verified,
mismatch, unrecorded or unverifiable — says whether this file is the one the
manifest binds and names this workspace's graph; it is the required relation.
Its *origin* — `files` or `synthetic` — says what produced the graph; it is a
description and never a binding state. A synthetic workspace is fully bound and
passes every check, and is shown as synthetic wherever it is shown.

Module: src/kg/provenance.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.file_modes import create_staging_file

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
#: `incomplete_by_design` among them: every writer since the record existed has
#: written it, and a record without it would be shown as establishing everything.
REQUIRED_RECORD_FIELDS = (
    "schema_version", "kg_digest", "origin", "sources", "incomplete_by_design",
)

#: What produced the graph: real input files, or a construction in memory (a
#: demo, a test, the deployment probe). A description, never a binding state.
ORIGINS = ("files", "synthetic")

#: Whether a record is the one its manifest binds and names this workspace's
#: graph (contract M3b). Only `verified` is accepted; the others say why not:
#: `mismatch`, the bytes or the graph named are another's; `unrecorded`, the
#: manifest binds no record; `unverifiable`, the record bound cannot be read.
BINDING_STATES = ("verified", "mismatch", "unrecorded", "unverifiable")

_IS_SHA256 = re.compile(r"[0-9a-f]{64}")


class ProvenanceError(ValueError):
    """A provenance record that cannot be trusted to describe its graph."""


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
    # The form `hashlib` gives, as for `kg_digest`: anything else can never be
    # the digest of the file it names, so it identifies nothing.
    if not (isinstance(digest, str) and _IS_SHA256.fullmatch(digest)):
        raise ProvenanceError(
            f"the digest recorded for {role} is not a SHA-256 hexdigest ({digest!r})"
        )
    if declared_version is not None and not isinstance(declared_version, str):
        raise ProvenanceError(
            f"{role} declares a version that is not a string ({declared_version!r})"
        )
    if path is not None and not isinstance(path, (str, os.PathLike)):
        raise ProvenanceError(f"{role} names its file as {path!r}, which is not a path")
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
    # **Entries are normalised here, not trusted because `source_entry` exists.**
    # Nothing stops a caller assembling a dict by hand: one with no digest
    # identifies nothing, and one with a directory in its filename or keys of its
    # own would carry them into every surface that shows the record. What is
    # stored is what `source_entry` builds.
    entries = [_normalised_entry(entry) for entry in (sources or [])]
    roles = {entry["role"] for entry in entries}
    missing = [role for role in SOURCE_ROLES if role not in roles]
    record = {
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
    # **The reader's rules, and only those, applied before anything is written**
    # (contract M3b): the origin, the graph digest, a synthetic record with
    # sources. A record every consumer would refuse is not produced; the
    # workspace writer runs this before its first byte, so the refusal lands at
    # the build, not at a training run an hour later.
    validate_record(record)
    require_identified(record)
    return record


def _normalised_entry(entry: Any) -> Dict[str, Any]:
    """A source entry as `source_entry` builds it: its role, its file's basename,
    its digest and its declared version, and nothing else."""
    if not isinstance(entry, dict):
        raise ProvenanceError(f"a source entry is not an object ({entry!r})")
    absent = [key for key in ("role", "digest") if key not in entry]
    if absent:
        raise ProvenanceError(
            f"a source entry is missing {absent}; without them it names "
            f"nothing and identifies nothing ({entry!r})"
        )
    return source_entry(
        role=entry["role"],
        path=entry.get("filename"),
        digest=entry["digest"],
        declared_version=entry.get("declared_version"),
    )


def validate_record(record: Any, where: str = "the provenance record") -> Dict[str, Any]:
    """The record's own rules, with no file around it (contract M3b).

    **One rule, two boundaries.** `build_provenance` and `write_provenance`
    apply it to what they are about to write, and the reader
    (`src.kg.artifacts.verify_dataset_record`) to what it parsed, so the writer
    cannot produce a record the reader refuses. `where` names the file when
    there is one.

    What a record must be: a JSON object of this schema (the integer 1), with
    its required fields; a `kg_digest` that is a SHA-256 hexdigest; an origin
    from `ORIGINS`; a list of sources, each exactly what `source_entry` builds
    (its role, its file's basename, a SHA-256 hexdigest, its declared version),
    no role twice; none for a synthetic origin; where it lists `missing_roles`,
    the roles its sources leave out (every role, for a synthetic one) — a record
    that contradicts itself is not read as either of its claims; and
    `incomplete_by_design`, a list of statements.

    Whether a `files` record identifies every input is not a shape rule: such a
    record is well formed and refused for what it says (`require_identified`).

    Raises:
        ProvenanceError: naming `where` and what was wrong.
    """
    if not isinstance(record, dict):
        raise ProvenanceError(f"{where} is not a JSON object, so it describes no build")
    version = record.get("schema_version")
    if (isinstance(version, bool) or not isinstance(version, int)
            or version != PROVENANCE_SCHEMA_VERSION):
        raise ProvenanceError(
            f"{where} is provenance schema {version!r}; this code reads "
            f"{PROVENANCE_SCHEMA_VERSION}. Which of its fields still mean what "
            "they say is a guess."
        )
    absent = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if absent:
        raise ProvenanceError(
            f"{where} is missing {absent}. A file that parses is not a record: "
            "without them it states no origin and lists no sources, which reads "
            "as a build with no inputs rather than as inputs unrecorded."
        )
    kg_digest = record["kg_digest"]
    if not (isinstance(kg_digest, str) and _IS_SHA256.fullmatch(kg_digest)):
        raise ProvenanceError(
            f"{where} records kg_digest {kg_digest!r}, which is not a SHA-256 "
            "hexdigest, so the record cannot say which graph it describes"
        )
    origin = record["origin"]
    if not isinstance(origin, str) or origin not in ORIGINS:
        raise ProvenanceError(f"{where} records origin {origin!r}, not one of {list(ORIGINS)}")
    sources = record["sources"]
    if not isinstance(sources, list):
        raise ProvenanceError(f"{where} records sources as {type(sources).__name__}, not a list")
    roles: List[str] = []
    for entry in sources:
        try:
            normalised = _normalised_entry(entry)
        except ProvenanceError as exc:
            raise ProvenanceError(f"{where}: {exc}") from exc
        if entry != normalised:
            raise ProvenanceError(
                f"{where} lists a source not in the form a build writes ({entry!r}): "
                "its role, its file's basename, its digest and its declared version, "
                "and nothing else"
            )
        roles.append(entry["role"])
    repeated = sorted({role for role in roles if roles.count(role) > 1})
    if repeated:
        raise ProvenanceError(
            f"{where} lists {repeated} more than once, so which file was that "
            "input is not said"
        )
    if origin == "synthetic" and sources:
        raise ProvenanceError(
            f"{where} is synthetic and lists source files. Nothing read them, so "
            "recording them would attribute a graph to inputs it never had."
        )
    expected = (
        [role for role in SOURCE_ROLES if role not in roles]
        if origin == "files" else list(SOURCE_ROLES)
    )
    if "missing_roles" in record and record["missing_roles"] != expected:
        raise ProvenanceError(
            f"{where} lists missing_roles {record['missing_roles']!r} and its "
            f"sources leave out {expected!r}; a record that contradicts itself "
            "is not read as either claim"
        )
    described = record["incomplete_by_design"]
    if not (isinstance(described, list) and all(isinstance(item, str) for item in described)):
        raise ProvenanceError(
            f"{where} records incomplete_by_design as {described!r}, not a list of statements"
        )
    return record


def unidentified_roles(record: Dict[str, Any]) -> List[str]:
    """The inputs a `files` record leaves unidentified, in `SOURCE_ROLES` order.

    **Every role is an input a real build requires**: the builder stops without
    either annotation file and always loads both ontologies
    (`scripts/build_knowledge_graph.py`). So a `files` record without one means
    a required input went unidentified.

    A synthetic record lists every role as missing by construction; that is its
    origin, not an omission, and this returns nothing for it.
    """
    if record.get("origin") != "files":
        return []
    present = {entry.get("role") for entry in record.get("sources", [])}
    return [role for role in SOURCE_ROLES if role not in present]


def require_identified(record: Dict[str, Any], where: str = "the provenance record") -> None:
    """Refuse a `files` record that leaves a required input unidentified
    (contract M3b). The check is not relaxed for a build that could not name an
    input: the builder identifies every input it parses, or stops."""
    missing = unidentified_roles(record)
    if missing:
        raise ProvenanceError(
            f"{where} is from files and leaves {missing} unidentified. Every real "
            f"build consumes all of {list(SOURCE_ROLES)}, so a required input went "
            "unidentified, and training, serving and measurement refuse such a "
            "record. Rebuild with scripts/build_knowledge_graph.py."
        )


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

    Returns the SHA-256 of the bytes written, so the caller binds it into a
    manifest without re-reading the file (contract M3b; BACKLOG 19.16). Same
    whole-then-replace shape as the split manifest, and for the same reason: a
    value no encoder takes must not leave a truncated file that parses as
    nothing and reads as a record that exists.

    The reader's rules are applied first (`validate_record`,
    `require_identified`): a record every consumer refuses is not written.
    """
    from src.utils.fingerprint import DigestingFile

    validate_record(record)
    require_identified(record)
    payload = encode_provenance(record).encode("utf-8")
    target = Path(workspace) / PROVENANCE_FILENAME
    handle, staged = create_staging_file(target)
    try:
        with handle:
            sink = DigestingFile(handle)
            sink.write(payload)
            sink.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return sink.sha256


__all__ = [
    "BINDING_STATES",
    "INCOMPLETE_BY_DESIGN",
    "ORIGINS",
    "PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceError",
    "REQUIRED_RECORD_FIELDS",
    "SOURCE_ROLES",
    "build_provenance",
    "encode_provenance",
    "require_identified",
    "source_entry",
    "unidentified_roles",
    "validate_record",
    "write_provenance",
]
