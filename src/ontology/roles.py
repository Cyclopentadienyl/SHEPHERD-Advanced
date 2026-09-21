"""
Is this the ontology the slot asked for.
========================================
`PLAN_ONTOLOGY_PHASE2.md` §3.4, and the defect that section was written from —
reproduced on this tree before it was designed against:

    hpo.obo  — declares `ontology: mondo`, contains two MONDO terms
      OntologyLoader.load()                  -> LOADED, name='mondo', 2 terms
      builder.add_ontology(ont, PHENOTYPE)   -> 0 nodes, no exception

The build carries on. `add_ontology` selects terms by the prefix belonging to
the node type, finds none, and returns zero; the provenance entry records role
`hpo` **because the caller passed that role**, with a faithful digest and a
faithful `data-version`. The result is a workspace with a complete-looking
record and no phenotype nodes.

**Provenance records what was opened; it does not decide what should have
been.** That is not a criticism of Phase 1 — a digest is an identity, not a
judgement — it is the reason this check has to exist somewhere else.

**Three things this is not.** It is not version negotiation: an old release
passes. It is not authenticity: a digest is not a publisher's signature, and
nothing here says a file came from who it claims. And it is not "every term
carries one prefix" — a legitimate ontology cross-references other namespaces,
which is exactly why `add_ontology` skips foreign terms rather than refusing
them, and a check that forbade them would reject real MONDO.

Module: src/ontology/roles.py

Dependencies: the standard library and `src.ontology.resolver`. No torch, no
pronto, no network.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.ontology.resolver import canonical_ontology_name

__all__ = [
    "ONTOLOGY_TERM_PREFIXES",
    "OntologyRoleError",
    "check_ontology_role",
    "term_prefix",
]

#: Which term-id prefix belongs to which ontology. **One home**, because
#: `KnowledgeGraphBuilder.add_ontology` needs the same association keyed by
#: `NodeType` and a second copy is how two tables come to disagree about what
#: `HP:` means.
ONTOLOGY_TERM_PREFIXES: Dict[str, str] = {
    "mondo": "MONDO:",
    "hpo": "HP:",
    "go": "GO:",
    "mp": "MP:",
}


def term_prefix(ontology: Any) -> Optional[str]:
    """The prefix for an ontology, or None when this project does not know it.

    None rather than a guess: an ontology with no known prefix cannot have its
    role checked by counting terms, and saying so is better than inventing a
    prefix that matches nothing and then refusing every file.
    """
    name = canonical_ontology_name(ontology)
    if name is None:
        return None
    return ONTOLOGY_TERM_PREFIXES.get(name)


class OntologyRoleError(ValueError):
    """The file is not the ontology this slot asked for."""


def check_ontology_role(ontology: Any, expected: str, *, source: Any = None) -> None:
    """Refuse a file that is not the ontology it is being used as.

    Two grounds, and they are different questions:

    1. **It declares another ontology.** A file whose header says `mondo` is
       not the HPO input whatever it is named. A file that declares *nothing*
       is not refused on this ground — absent is not contradictory, and OBO
       files without an `ontology:` tag exist.
    2. **The slot's namespace yields no usable term.** This is the one that
       catches a file declaring nothing, and it is measured against the same
       prefix `add_ontology` will select by — so "this passes and then builds
       zero nodes" is not a reachable state.

    Args:
        ontology: a loaded `src.ontology.hierarchy.Ontology`.
        expected: the canonical name of the slot, e.g. `"hpo"`.
        source: the file, named in any refusal.

    Raises:
        OntologyRoleError: naming the file, what it declares and what was
            expected — the three things an operator needs to fix it.
    """
    wanted = canonical_ontology_name(expected) or str(expected)
    where = f"{source}" if source is not None else "the ontology supplied"

    declared_raw = getattr(ontology, "name", None)
    declared = canonical_ontology_name(declared_raw)
    # `Ontology.name` falls back to the literal "Unknown" when a file declares
    # nothing. Treating that as a declaration would refuse every undeclared
    # file on ground 1 and never reach ground 2, which is the check that
    # actually protects the build.
    if isinstance(declared_raw, str) and declared_raw.strip().lower() == "unknown":
        declared = None

    if declared is not None and declared != wanted:
        raise OntologyRoleError(
            f"{where} declares itself to be {declared_raw!r}, and it is being "
            f"used as the {wanted} input. Node indices are assigned in the "
            "order terms are inserted, so building from the wrong ontology "
            "does not produce a wrong label — it produces a workspace whose "
            f"{wanted} nodes are missing entirely, with a provenance record "
            "that looks complete. Pass the right file, or the right slot."
        )

    prefix = term_prefix(wanted)
    if prefix is None:
        # Nothing to count against. The declaration check above is all this
        # project can say about an ontology it has no prefix for, and a
        # refusal here would be a refusal of the unknown rather than of a
        # mismatch.
        return

    try:
        terms = ontology.get_all_terms(include_obsolete=False)
    except Exception as exc:  # pragma: no cover - defensive, shape-dependent
        raise OntologyRoleError(
            f"{where} could not be inspected for {wanted} terms "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    # **Any, not all.** A real ontology cross-references other namespaces and
    # `add_ontology` skips those by design; requiring every term to carry the
    # prefix would refuse genuine MONDO.
    if not any(isinstance(term, str) and term.startswith(prefix) for term in terms):
        total = len(terms)
        raise OntologyRoleError(
            f"{where} carries no {prefix} terms, so as the {wanted} input it "
            f"would contribute no nodes at all ({total} terms present, none of "
            f"them {prefix}). A build would succeed, record this file's digest "
            "and version, and produce a workspace missing an entire node type."
        )
