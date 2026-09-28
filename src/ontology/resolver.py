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
    "IMPORT_WITHOUT_TARGET",
    "NoOntologyCandidateError",
    "OntologyCandidate",
    "OntologyResolutionError",
    "canonical_ontology_name",
    "declared_imports",
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

#: OBO Foundry publishes each ontology in several release products —
#: `mondo-base`, `hp-simple`, `mondo-edit` and so on — and they declare
#: themselves by that product name (`ontology: mondo/mondo-base`). **A base
#: release of MONDO is MONDO.** Treating it as another ontology made a staged
#: variant vanish from its own candidate list without a word, after which the
#: build downloaded and used a different file; and an explicitly named variant
#: was refused as "the wrong ontology" by a message claiming it would build zero
#: nodes, while the file carried the right terms. Which product it was is what
#: the digest and `data-version` record; the role check's term test still
#: guards the content.
_RELEASE_VARIANTS = (
    "non-classified", "international", "simple", "basic", "base", "full",
    "plus", "edit",
)


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
    # An IRI fragment names a part of the ontology, not the ontology. Without
    # this, `…/mondo.owl#` reduced to `mondo.owl#` and the file was dropped from
    # its own candidate list.
    key = key.split("#", 1)[0]
    if "/" in key or ":" in key:
        # An IRI, a URL or a versioned path. Trailing slashes first, so that
        # `.../mondo.owl/` does not reduce to the empty string.
        key = key.rstrip("/").rsplit("/", 1)[-1]
    if not key:
        return None
    for suffix in ONTOLOGY_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    if key in _ALIASES:
        return _ALIASES[key]
    # **Products stack.** HPO publishes `hp-simple-non-classified`, declared as
    # `ontology: hp/hp-simple-non-classified`, so one suffix is not the rule.
    stem = key
    while True:
        for variant in _RELEASE_VARIANTS:
            if stem.endswith("-" + variant):
                stem = stem[: -(len(variant) + 1)]
                break
        else:
            break
    if stem in _ALIASES:
        return _ALIASES[stem]
    return key


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
        """One line for a refusal an operator has to act on.

        **Including the declared imports**, which §3.2 says a listing shows.
        Without them an operator choosing between two candidates could pick the
        one that is certain to be refused at load and learn that only then.
        """
        version = self.data_version or "no data-version declared"
        line = (
            f"{self.path} — {version}, {self.term_count} "
            f"{self.term_count_basis}, sha256 {self.digest[:12]}..."
        )
        if self.declared_imports:
            line += (
                f" — DECLARES {len(self.declared_imports)} IMPORT(S) "
                f"({', '.join(self.declared_imports)}), so it will be refused at load"
            )
        return line


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


#: What an import declaration with no readable target is recorded as.
#: `<owl:imports/>`, `rdf:resource=""`, a blank-node `rdf:nodeID`, or an OBO
#: `import:` with no value all **declare** a dependency; not being able to say
#: which one is no reason to treat the file as self-contained. pronto records
#: the same case as `None`, and the loader reports it under this label too.
IMPORT_WITHOUT_TARGET = "(an import with no target)"


class _OwlIdentity:
    """An ElementTree parser *target* that keeps counts and nothing else.

    Given to `XMLParser(target=...)`, it receives each start tag with its
    namespace-expanded name and attributes, and the parser keeps no element
    once this method returns. Memory therefore depends on the deepest open
    element, not on how many there have been.

    **Named classes only.** `<owl:Class>` with no `rdf:about` is an anonymous
    class expression inside an axiom — a restriction or a union — not a term,
    and counting it made `term_count` larger than the ontology while its label
    said "owl:Class declarations".

    **An import is an import wherever RDF/XML puts it.** The usual form is
    `<owl:imports rdf:resource="…"/>` under the ontology element, but the same
    triple can be written inside an `rdf:Description` about the ontology, or as
    `<owl:imports><owl:Ontology rdf:about="…"/></owl:imports>`. The loader
    refuses on the union of this and pronto's own reading, so neither form is
    loaded with its import silently dropped.
    """

    def __init__(self) -> None:
        self.terms = 0
        self.imports: List[str] = []
        self.data_version: Optional[str] = None
        self.declared: Optional[str] = None
        self._depth = 0
        self._inside_imports = 0
        self._import_has_target = False

    def start(self, tag: str, attrib: Dict[str, str]) -> None:
        self._depth += 1
        if self._inside_imports:
            target = (attrib.get(_RDF_ABOUT) or attrib.get(_RDF_RESOURCE) or "").strip()
            if target:
                self.imports.append(target)
                self._import_has_target = True
            return
        if tag == _OWL_CLASS_TAG:
            if attrib.get(_RDF_ABOUT):
                self.terms += 1
        elif tag == _OWL_IMPORTS_TAG:
            # **A declaration is recorded whether or not it names a target.**
            # The first version recorded only a readable URL, so `<owl:imports/>`
            # inside an `rdf:Description` — which pronto does not see either —
            # left no trace anywhere and the file loaded as self-contained.
            target = (attrib.get(_RDF_RESOURCE) or "").strip()
            if target:
                self.imports.append(target)
            else:
                self._inside_imports = self._depth
                self._import_has_target = False
        elif tag == _OWL_VERSION_TAG and self.data_version is None:
            self.data_version = attrib.get(_RDF_RESOURCE) or None
        elif tag == _OWL_ONTOLOGY and self.declared is None:
            self.declared = attrib.get(_RDF_ABOUT) or None

    def end(self, tag: str) -> None:
        if self._inside_imports and self._depth == self._inside_imports:
            if not self._import_has_target:
                self.imports.append(IMPORT_WITHOUT_TARGET)
            self._inside_imports = 0
        self._depth -= 1

    def close(self) -> None:
        return None


#: How much of a file to look at before deciding what format it is in.
_SNIFF_BYTES = 4096

#: XML is read in blocks rather than lines: a document written without line
#: breaks is one "line", and reading it line by line held the whole file.
_XML_BLOCK = 1 << 16


def _sniff_is_xml(path: Path) -> bool:
    """**What the content is, not what the name says.**

    pronto decides the format from the bytes, so a resolver that decided from
    the suffix disagreed with the loader it feeds: an `.owl` holding OBO text
    was "not well-formed XML", dropped from the listing, and the build then
    downloaded a different file — or took the only other candidate as
    unambiguous. Whatever begins with `<` after a byte-order mark and white
    space is XML; everything else is read as OBO.
    """
    with open(path, "rb") as handle:
        return _head_is_xml(handle.read(_SNIFF_BYTES))


def _head_is_xml(head: bytes) -> bool:
    return head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<")


def _clean_obo_value(raw: str) -> str:
    """A header value without its trailing `! comment` or `{qualifiers}`.

    OBO allows both after any tag value, so `ontology: hp ! the HPO` declares
    `hp`. Keeping the comment made a loadable file declare a name nothing
    recognised, and it dropped out of its own candidate list. A backslash
    escapes a literal `!`.
    """
    out: List[str] = []
    escaped = False
    for char in raw:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            out.append(char)
            continue
        if char == "!":
            break
        out.append(char)
    value = "".join(out).strip()
    return re.sub(r"\s*\{[^{}]*\}\s*$", "", value).strip()


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
    try:
        is_obo = not _sniff_is_xml(path)
    except OSError as exc:
        raise OntologyResolutionError(
            f"{path} could not be read ({type(exc).__name__}: {exc})"
        ) from exc
    basis = "obo [Term] stanzas" if is_obo else "owl:Class declarations"
    in_header = True
    # **Fed the same bytes the digest sees, so it is still one pass.** And fed
    # to a parser that **builds no tree**: `_OwlIdentity` is an ElementTree
    # *target*, so each start tag is handed over and forgotten. The previous
    # version used `XMLPullParser`, whose default tree builder keeps every
    # element attached to its parent until the document ends — reading its
    # events empties the queue and frees nothing. Measured on a generated
    # RDF/XML with labels and subClassOf, the peak was about five times the
    # file (36 MB in, 182 MB held), and enumeration scans every candidate
    # before filtering, so a large MONDO OWL was built into memory even when
    # the build wanted HPO. The comment above that code said "bounded"; it
    # was not. Nothing here resolves anything either: expat fetches no
    # external entity and has no notion of `owl:imports`.
    owl = None if is_obo else _OwlIdentity()
    xml = None if is_obo else ElementTree.XMLParser(target=owl)

    try:
        with open(path, "rb") as handle:
            if is_obo:
                first = True
                for raw in handle:
                    # The digest is of the bytes on disk, byte-order mark
                    # included; only the copy that is parsed has it removed.
                    digest.update(raw)
                    line = raw.lstrip(b"\xef\xbb\xbf") if first else raw
                    first = False
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
                    value = _clean_obo_value(tag.group(2).decode("utf-8", "replace"))
                    if key == "data-version" and data_version is None:
                        data_version = value or None
                    elif key == "ontology" and declared is None:
                        declared = value or None
                    elif key == "import":
                        imports.append(value or IMPORT_WITHOUT_TARGET)
            else:
                while True:
                    block = handle.read(_XML_BLOCK)
                    if not block:
                        break
                    digest.update(block)
                    xml.feed(block)
                # **Close before trusting the counts.** A document that ends
                # mid-element has not been fully described, and reporting a
                # partial count as identity is how a truncated file comes to
                # look like a smaller release. `close` raises on it.
                xml.close()
                terms = owl.terms
                imports = owl.imports
                data_version = owl.data_version
                declared = owl.declared
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
            # Hidden files are not candidates: `._mondo.obo` is macOS metadata,
            # and a dotted copy is what tools leave while writing. (The loader's
            # own staging name, `.mondo.obo.staged`, is already excluded by its
            # suffix; this is the second reason, not the only one.)
            if entry.name.startswith("."):
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
                if name == wanted:
                    # **Not silent.** A file named for this ontology that
                    # declares another is exactly the misfiled case §3.4 is
                    # about, and an operator whose build then downloads a
                    # replacement deserves to know why the file they staged
                    # was passed over.
                    logger.warning(
                        "%s is named like a %s file but declares %r, so it is "
                        "not a %s candidate",
                        entry, wanted, declared, wanted,
                    )
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
                f"the explicitly named {wanted} file {path} does not exist. An "
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
            f"{_how_to_name(wanted)} They are not ranked by root "
            "order, by modification time or by data-version — a release is a "
            "decision, and a string a file declares about itself is not a "
            "basis for taking it automatically.",
            candidates,
        )

    chosen = candidates[0]
    logger.info(
        "%s: one candidate under the configured roots, %s (%s, sha256 %s)",
        wanted, chosen.path, chosen.data_version or "no data-version declared",
        chosen.digest,
    )
    return chosen


#: The ontologies the build CLI has a path flag for. The resolver is also
#: called from the library, where there is no such flag, and naming one there
#: sent a `load_go()` caller looking for `--go-path`, which does not exist.
_CLI_PATH_FLAGS = ("mondo", "hpo")


def _how_to_name(wanted: str) -> str:
    if wanted in _CLI_PATH_FLAGS:
        return (
            f"Name one with --{wanted}-path on the build CLI, or explicit_path= "
            "when calling select_ontology_file."
        )
    return "Name one with explicit_path= when calling select_ontology_file."


def declared_imports(source: Any) -> Tuple[str, ...]:
    """Every import a file declares, read the way the listing reads it.

    **One rule for the loader and the listing.** pronto records only
    `owl:imports` elements that are direct children of the first
    `owl:Ontology`; the listing records them wherever RDF/XML puts them. With
    the loader refusing on pronto's set alone, a file could be listed as
    "will be refused" and then load with its import silently dropped. The
    loader refuses on the union of the two.

    **A path or an open binary handle.** The loader passes the handle it
    hashed and parsed, so the imports it refuses on are read from the same
    bytes — not from whatever the path names by the time a second open
    happens. A handle is read from its current position.

    For OBO only the header is read, since that is the only place an `import:`
    tag may appear — so this costs a few kilobytes on a large MONDO, not a pass
    over it.
    """
    if not hasattr(source, "read"):
        with open(Path(source), "rb") as handle:
            return declared_imports(handle)
    handle = source
    start = handle.tell()
    head = handle.read(_SNIFF_BYTES)
    handle.seek(start)
    if _head_is_xml(head):
        owl = _OwlIdentity()
        xml = ElementTree.XMLParser(target=owl)
        try:
            while True:
                block = handle.read(_XML_BLOCK)
                if not block:
                    break
                xml.feed(block)
            xml.close()
        except ElementTree.ParseError as exc:
            raise OntologyResolutionError(
                f"{getattr(handle, 'name', 'the file')} is not well-formed XML ({exc})"
            ) from exc
        return tuple(owl.imports)
    found: List[str] = []
    first = True
    for raw in handle:
        line = raw.lstrip(b"\xef\xbb\xbf") if first else raw
        first = False
        if _OBO_STANZA.match(line):
            break
        tag = _OBO_TAG.match(line)
        if tag and tag.group(1).decode("utf-8", "replace").lower() == "import":
            value = _clean_obo_value(tag.group(2).decode("utf-8", "replace"))
            found.append(value or IMPORT_WITHOUT_TARGET)
    return tuple(found)
