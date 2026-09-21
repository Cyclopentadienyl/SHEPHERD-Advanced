"""
Which ontology file, out of the ones that are here.
===================================================
`PLAN_ONTOLOGY_PHASE2.md` §3.1 and §3.2. The build used to open
`<cache_dir>/<name>.obo` by convention, so Phase 1's provenance recorded what was
used for an input **nobody selected**. This module is the selection.

**One rule, and it is about ambiguity rather than about choice.** With no basis
for choosing, do not choose. An explicit path wins; exactly one candidate is
taken; more than one **refuses** and hands the operator each candidate's path,
`data-version` and digest so they can name the one they meant. Nothing picks by
root order, modification time, or "the newest `data-version`" — ordering by a
string a file declares about itself is a guess wearing a comparison, and a rule
that picks silently is a rule nobody reads.

What is *not* fixed by that: which ontology, which release, which directory, or
how many coexist. Naming a path automates a build completely, with no file
deleted and no code changed.

**Identity is read without a parser and without the network.** A count of terms
would normally mean a parse, and a parse at pronto's default resolves `import:`
lines over the network (`PLAN_ONTOLOGY_PHASE2.md` §1.5) — so enumeration would
reach the internet before anyone had chosen a file. Every field here comes from
one streaming pass over the bytes: the digest, the header tags, the declared
imports and the stanza count, in a single read, in bounded memory.

**Nothing here creates a directory.** `OntologyLoader.__init__` mkdirs its cache
(`loader.py:64`), so merely constructing one makes a directory appear. A resolver
reads roots; a root that is not there is a root with nothing in it.

Module: src/ontology/resolver.py

Dependencies: the standard library only. No torch, no pronto, no network.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from xml.etree import ElementTree
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AmbiguousOntologyError",
    "NoOntologyCandidateError",
    "OntologyCandidate",
    "OntologyResolutionError",
    "canonical_ontology_name",
    "enumerate_candidates",
    "scan_identity",
    "select_ontology_file",
]

#: Extensions a candidate can have, in no order of preference. **`.obo` is not
#: preferred over `.owl`** — the download fallback writes whichever format
#: succeeded, whenever it ran, so two files in one directory are not two
#: encodings of one release unless their `data-version` says so.
ONTOLOGY_SUFFIXES = (".obo", ".owl")

#: Spellings of one ontology. `hp.obo` and `hpo.obo` are the same ontology, and
#: a resolver that treated them as two would refuse a directory holding one file
#: under each name.
_ALIASES = {
    "hp": "hpo",
    "human_phenotype": "hpo",
    "mondo": "mondo",
    "hpo": "hpo",
    "go": "go",
    "mp": "mp",
}


def canonical_ontology_name(name: Any) -> Optional[str]:
    """`hp` and `hpo` are one ontology; anything unrecognised is itself.

    **An IRI is a name too**, and that is not a nicety: OBO files declare
    `ontology: mondo` while RDF/XML declares
    `<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/mondo.owl">`. A
    comparison that handled only the first read every OWL file as declaring
    something other than itself — which filtered them out of their own
    ontology's candidate list, so a directory holding an `.obo` and an `.owl`
    resolved to one candidate and was taken silently. The last path segment is
    what carries the name.

    Returns None for a value that is not a usable name at all, so a caller can
    tell "this file declares nothing" from "this file declares something else".
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    if not key:
        return None
    if "/" in key or ":" in key:
        # An IRI, a URL or a versioned path. Trailing slashes first, so that
        # `.../mondo.owl/` does not reduce to the empty string.
        key = key.rstrip("/").rsplit("/", 1)[-1]
        if not key:
            return None
    for suffix in ONTOLOGY_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return _ALIASES.get(key, key)


class OntologyResolutionError(ValueError):
    """Selection could not be completed. Never a silent fallback."""


class AmbiguousOntologyError(OntologyResolutionError):
    """More than one candidate and no basis for choosing between them."""

    def __init__(self, message: str, candidates: Sequence["OntologyCandidate"]):
        super().__init__(message)
        self.candidates = tuple(candidates)


class NoOntologyCandidateError(OntologyResolutionError):
    """No candidate under any root.

    **Its own type because it is the one outcome that is not a refusal of the
    workspace.** The caller may download (`PLAN_ONTOLOGY_PHASE2.md` §3.1) — but
    it has to decide that deliberately, which it cannot do if this arrives as
    the same exception as an ambiguous directory.
    """

    def __init__(self, message: str, roots: Sequence[Path]):
        super().__init__(message)
        self.roots = tuple(roots)


@dataclass(frozen=True)
class OntologyCandidate:
    """One ontology file, and what can be known about it without parsing it.

    `declared_ontology` is what the file says it is, which is not what the
    filename says it is — the two disagreeing is the defect `§3.4` refuses.
    `declared_imports` non-empty means the imports policy will refuse this file
    at load; it is carried here so a listing can say so before anyone chooses.
    """

    ontology: str
    path: Path
    root: Optional[Path]
    digest: str
    size_bytes: int
    data_version: Optional[str]
    declared_ontology: Optional[str]
    declared_imports: Tuple[str, ...]
    term_count: int
    term_count_basis: str

    def describe(self) -> str:
        """One line for a refusal an operator has to act on."""
        version = self.data_version or "no data-version declared"
        return (
            f"{self.path} — {version}, {self.term_count} "
            f"{self.term_count_basis}, sha256 {self.digest[:12]}..."
        )


# --- Identity, from one pass over the bytes -------------------------------

#: An OBO header tag ends at the first stanza. Matched on bytes, because the
#: same pass feeds the digest and a decode of a multi-gigabyte file to find four
#: tags would be a second representation of the whole thing.
_OBO_STANZA = re.compile(rb"^\[[^\]]*\]\s*$")
_OBO_TERM = re.compile(rb"^\[Term\]\s*$")
_OBO_TAG = re.compile(rb"^([A-Za-z_-]+):[ \t]*(.*?)\s*$")

#: RDF/XML is read as XML, **by namespace rather than by prefix**. The first
#: version matched the literal strings `owl:Class` and `rdf:about` line by
#: line, which is wrong twice over: a prefix is chosen by the document
#: (`xmlns:w="…owl#"` is as valid as `xmlns:owl=`), and an attribute may sit on
#: a different line from its element. Both shapes made a real, loadable OWL
#: file report no ontology and no terms — so it was filtered out of its own
#: candidate list, and a directory holding it beside an `.obo` resolved to one
#: candidate and was taken silently. That is the ambiguity refusal defeated by
#: a scanner, which is worse than not listing the file at all.
_OWL_NS = "http://www.w3.org/2002/07/owl#"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_OWL_ONTOLOGY = f"{{{_OWL_NS}}}Ontology"
_OWL_CLASS_TAG = f"{{{_OWL_NS}}}Class"
_OWL_VERSION_TAG = f"{{{_OWL_NS}}}versionIRI"
_OWL_IMPORTS_TAG = f"{{{_OWL_NS}}}imports"
_RDF_ABOUT = f"{{{_RDF_NS}}}about"
_RDF_RESOURCE = f"{{{_RDF_NS}}}resource"


def scan_identity(path: Any) -> Dict[str, Any]:
    """Digest, size and declared identity, in a single read.

    **One pass, deliberately.** Hashing the file and then parsing it are two
    reads of the same bytes, and between them a publisher can finish writing —
    the shortest-path artifact round had exactly that defect in a different
    file. Reading line by line in binary keeps the memory bounded and the two
    answers consistent with each other.

    Raises:
        OntologyResolutionError: the path is not a readable regular file.
    """
    path = Path(path)
    try:
        if not path.is_file():
            raise OntologyResolutionError(
                f"{path} is not a regular file, so it is not an ontology to read"
            )
        size = path.stat().st_size
    except OSError as exc:
        raise OntologyResolutionError(
            f"{path} could not be examined ({type(exc).__name__}: {exc})"
        ) from exc

    digest = hashlib.sha256()
    data_version: Optional[str] = None
    declared: Optional[str] = None
    imports: List[str] = []
    terms = 0
    basis = "obo [Term] stanzas" if path.suffix.lower() == ".obo" else "owl:Class declarations"
    is_obo = path.suffix.lower() == ".obo"
    in_header = True
    # **Fed the same bytes the digest sees, so it is still one pass.** An
    # incremental parser keeps the memory bounded and, unlike pronto, resolves
    # nothing: `ElementTree` has no notion of `owl:imports` and issues no
    # request for one.
    xml = None if is_obo else ElementTree.XMLPullParser(events=("start",))

    try:
        with open(path, "rb") as handle:
            for line in handle:
                digest.update(line)
                if is_obo:
                    if _OBO_TERM.match(line):
                        terms += 1
                        in_header = False
                        continue
                    if _OBO_STANZA.match(line):
                        in_header = False
                        continue
                    if not in_header:
                        continue
                    tag = _OBO_TAG.match(line)
                    if tag is None:
                        continue
                    key = tag.group(1).decode("utf-8", "replace").lower()
                    value = tag.group(2).decode("utf-8", "replace")
                    if key == "data-version" and data_version is None:
                        data_version = value or None
                    elif key == "ontology" and declared is None:
                        declared = value or None
                    elif key == "import" and value:
                        imports.append(value)
                else:
                    xml.feed(line)
                    for _, element in xml.read_events():
                        tag = element.tag
                        if tag == _OWL_CLASS_TAG:
                            terms += 1
                        elif tag == _OWL_IMPORTS_TAG:
                            target = element.get(_RDF_RESOURCE)
                            if target:
                                imports.append(target)
                        elif tag == _OWL_VERSION_TAG and data_version is None:
                            data_version = element.get(_RDF_RESOURCE) or None
                        elif tag == _OWL_ONTOLOGY and declared is None:
                            declared = element.get(_RDF_ABOUT) or None
        if xml is not None:
            # **Close before trusting the counts.** A document that ends
            # mid-element has not been fully described, and reporting a partial
            # count as identity is how a truncated file comes to look like a
            # smaller release.
            xml.close()
            for _, element in xml.read_events():
                if element.tag == _OWL_CLASS_TAG:
                    terms += 1
    except OSError as exc:
        raise OntologyResolutionError(
            f"{path} could not be read ({type(exc).__name__}: {exc})"
        ) from exc
    except ElementTree.ParseError as exc:
        raise OntologyResolutionError(
            f"{path} is not well-formed XML ({exc}); it cannot be identified as "
            "an ontology file, so it is neither listed nor selected"
        ) from exc

    return {
        "digest": digest.hexdigest(),
        "size_bytes": size,
        "data_version": data_version,
        "declared_ontology": declared,
        "declared_imports": tuple(imports),
        "term_count": terms,
        "term_count_basis": basis,
    }


def _candidate(path: Path, ontology: str, root: Optional[Path]) -> OntologyCandidate:
    return OntologyCandidate(ontology=ontology, path=path, root=root, **scan_identity(path))


def _name_from_path(path: Path) -> Optional[str]:
    return canonical_ontology_name(path.stem)


def _claims_ontology(candidate_name: Optional[str], declared: Optional[str], wanted: str) -> bool:
    """Does this file belong to the ontology being asked for?

    **The declaration wins where there is one**, because the filename is a
    convention and the header is the file's own statement. A file named
    `hpo.obo` that declares `mondo` is therefore not an HPO candidate — it is
    the §3.4 mismatch, and listing it as an HPO candidate would turn a clear
    refusal into an ambiguous one.
    """
    declared_name = canonical_ontology_name(declared)
    if declared_name is not None:
        return declared_name == wanted
    return candidate_name == wanted


def enumerate_candidates(
    roots: Iterable[Any],
    ontology: Optional[str] = None,
) -> List[OntologyCandidate]:
    """Every ontology file under every root, with its identity.

    Roots are read in the order given **for listing only**; the order carries no
    precedence and `select_ontology_file` does not consult it. A root that does
    not exist contributes nothing and is not created.

    A file that cannot be read is logged and skipped rather than raising: one
    unreadable file in a directory of many is not a reason to refuse to describe
    the others, and the selection below refuses on its own terms.
    """
    wanted = canonical_ontology_name(ontology) if ontology is not None else None
    found: List[OntologyCandidate] = []
    seen: set = set()

    for raw_root in roots:
        root = Path(raw_root)
        try:
            if not root.is_dir():
                continue
            entries = sorted(root.iterdir())
        except OSError as exc:
            logger.warning("ontology root %s could not be listed (%s)", root, exc)
            continue

        for entry in entries:
            if entry.suffix.lower() not in ONTOLOGY_SUFFIXES:
                continue
            try:
                resolved = entry.resolve()
            except OSError:
                resolved = entry
            if resolved in seen:
                continue
            name = _name_from_path(entry)
            try:
                identity = scan_identity(entry)
            except OntologyResolutionError as exc:
                logger.warning("skipping %s: %s", entry, exc)
                continue
            declared = identity["declared_ontology"]
            if wanted is not None and not _claims_ontology(name, declared, wanted):
                continue
            seen.add(resolved)
            found.append(
                OntologyCandidate(
                    ontology=wanted or canonical_ontology_name(declared) or name or entry.stem,
                    path=entry,
                    root=root,
                    **identity,
                )
            )
    return found


def select_ontology_file(
    ontology: str,
    *,
    roots: Iterable[Any] = (),
    explicit_path: Any = None,
) -> OntologyCandidate:
    """**The one place a list becomes a choice.**

    Every surface goes through here: the build CLI passes a path, and a later
    interface lists `enumerate_candidates` and passes the one a person picked.
    A refusal buried inside the loader would make that interface unusable for
    the very case it exists to handle.

    Args:
        ontology: the canonical name being selected for, e.g. `"mondo"`.
        roots: directories to search when no path is given. Order is not
            precedence.
        explicit_path: a file named by the caller. Wins outright.

    Returns:
        The selected candidate, with its identity already read.

    Raises:
        OntologyResolutionError: an explicit path that is not a readable file.
        AmbiguousOntologyError: more than one candidate and nothing to choose by.
        NoOntologyCandidateError: none at all — the caller may download.
    """
    wanted = canonical_ontology_name(ontology) or str(ontology)

    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise OntologyResolutionError(
                f"--{wanted}-path names {path}, which does not exist. An "
                "explicitly named file is not a hint: nothing falls back to a "
                "configured root or to the ontology cache, because a build that "
                "quietly used a different file than the one asked for is the "
                "failure this selection exists to remove."
            )
        candidate = _candidate(path, wanted, root=None)
        logger.info(
            "%s: using the explicitly named %s (%s)",
            wanted, path, candidate.data_version or "no data-version declared",
        )
        return candidate

    roots = [Path(root) for root in roots]
    candidates = enumerate_candidates(roots, ontology=wanted)

    if not candidates:
        raise NoOntologyCandidateError(
            f"no {wanted} ontology file found under "
            f"{', '.join(str(root) for root in roots) or '(no roots configured)'}",
            roots,
        )

    if len(candidates) > 1:
        listing = "\n".join(f"  {item.describe()}" for item in candidates)
        raise AmbiguousOntologyError(
            f"{len(candidates)} {wanted} ontology files are available and "
            f"nothing says which one this build should use:\n{listing}\n"
            f"Pass --{wanted}-path to name one. They are not ranked by root "
            "order, by modification time or by data-version — a release is a "
            "decision, and a string a file declares about itself is not a "
            "basis for taking it automatically.",
            candidates,
        )

    chosen = candidates[0]
    logger.info(
        "%s: one candidate under the configured roots, %s (%s)",
        wanted, chosen.path, chosen.data_version or "no data-version declared",
    )
    return chosen
