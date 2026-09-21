"""Two refusals the loader owes: a file that reaches out, and a file in the wrong slot.

`PLAN_ONTOLOGY_PHASE2.md` §3.3 and §3.4; acceptance 14-20.

**The imports case is a trap rather than an omission.** Setting pronto's
`import_depth` to 0 is the obvious way to stop a parse fetching over the
network, and measured it is *worse* than the default: the file loads, the import
is dropped, `imports` comes back empty and nothing warns. A build would then be
missing whatever the import carried while looking complete. `metadata.imports`
keeps the declaration at depth 0, and refusing on that is the policy.

**The role case was reproduced before it was designed against.** A file named
`hpo.obo` declaring `ontology: mondo` loaded, built 0 phenotype nodes without
raising, and was recorded in provenance as the `hpo` input with a faithful
digest — because the caller passed that role and a digest is an identity, not a
judgement.

Module: tests/unit/test_ontology_imports_and_role.py
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ontology.loader import OntologyImportError, OntologyLoader  # noqa: E402
from src.ontology.roles import (  # noqa: E402
    ONTOLOGY_TERM_PREFIXES,
    OntologyRoleError,
    check_ontology_role,
    term_prefix,
)


def obo(path: Path, *, ontology="hpo", terms=("HP:0000001",), imports=(),
        version="releases/2026-09-01") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["format-version: 1.2"]
    if version:
        lines.append(f"data-version: {version}")
    if ontology:
        lines.append(f"ontology: {ontology}")
    lines += [f"import: {item}" for item in imports]
    for term in terms:
        lines += ["", "[Term]", f"id: {term}", f"name: term {term}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def loader(tmp_path) -> OntologyLoader:
    return OntologyLoader(cache_dir=tmp_path / "cache")


class TestTheImportsPolicy:

    def test_a_declared_import_is_refused_and_named(self, tmp_path):
        """**Acceptance 14.** It must not load with the import dropped."""
        path = obo(tmp_path / "hpo.obo", imports=("http://purl.obolibrary.org/obo/x.obo",))

        with pytest.raises(OntologyImportError) as caught:
            loader(tmp_path).load(path)

        assert "http://purl.obolibrary.org/obo/x.obo" in str(caught.value)

    def test_an_import_pointing_at_a_local_file_is_refused_too(self, tmp_path):
        """**Acceptance 15.** §3.3 as written, not as the first draft said —
        "cannot be satisfied locally" is a weaker rule than the policy
        implements, and a local import is still content the root file's digest
        does not cover."""
        sibling = obo(tmp_path / "extra.obo", ontology="mondo", terms=("MONDO:1",))
        path = obo(tmp_path / "hpo.obo", imports=(sibling.name,))

        with pytest.raises(OntologyImportError, match="self-contained"):
            loader(tmp_path).load(path)

    def test_a_self_contained_file_loads(self, tmp_path):
        """**Acceptance 16.** Without this, the refusals above are satisfied by
        a loader that rejects everything."""
        path = obo(tmp_path / "hpo.obo", terms=("HP:1", "HP:2"))

        ontology = loader(tmp_path).load(path)

        assert ontology.num_terms == 2

    def test_nothing_is_fetched_while_parsing(self, tmp_path, monkeypatch):
        """The refusal has to come from the declaration, not from a failed
        fetch — otherwise a *reachable* import would be resolved and silently
        folded into the graph, which is the case a test using an unreachable
        host would never see."""
        def explode(*a, **k):
            raise AssertionError("the parse reached the network")

        for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
            monkeypatch.setattr(socket, name, explode, raising=False)

        path = obo(tmp_path / "hpo.obo", imports=("http://purl.obolibrary.org/obo/x.obo",))

        with pytest.raises(OntologyImportError):
            loader(tmp_path).load(path)

    def test_a_clean_file_parses_without_touching_the_network_either(self, tmp_path):
        """`import_depth` is a constant on the class, so no call site can
        reintroduce pronto's unbounded default by omitting the argument."""
        assert OntologyLoader.IMPORT_DEPTH == 0

    def test_the_refusal_names_every_import_not_just_the_first(self, tmp_path):
        path = obo(tmp_path / "hpo.obo",
                   imports=("http://a.invalid/1.obo", "http://b.invalid/2.obo"))

        with pytest.raises(OntologyImportError) as caught:
            loader(tmp_path).load(path)

        assert "1.obo" in str(caught.value) and "2.obo" in str(caught.value)


class TestTheRoleCheck:

    def test_the_reproduction_is_refused(self, tmp_path):
        """**Acceptance 17.** The exact file from §3.4: named `hpo.obo`,
        declaring `mondo`, carrying MONDO terms. It used to load and build 0
        phenotype nodes with no exception anywhere."""
        path = obo(tmp_path / "hpo.obo", ontology="mondo",
                   terms=("MONDO:0000001", "MONDO:0000002"))

        with pytest.raises(OntologyRoleError) as caught:
            loader(tmp_path).load(path, expect="hpo")

        message = str(caught.value)
        assert "mondo" in message and "hpo" in message and str(path) in message

    def test_without_the_check_it_still_loads(self, tmp_path):
        """The check is opt-in at the library level and the build passes it.
        Stated here so the next reader knows `load(path)` alone is not the
        gate — the CLI's use of `expect` is."""
        path = obo(tmp_path / "hpo.obo", ontology="mondo", terms=("MONDO:1",))

        assert loader(tmp_path).load(path).num_terms == 1

    def test_a_file_carrying_no_terms_of_the_slot_is_refused(self, tmp_path):
        """The ground that catches a file declaring nothing at all — the
        declaration check cannot, and this is the one that actually protects
        the build."""
        path = obo(tmp_path / "anything.obo", ontology=None, terms=("MONDO:1",))

        with pytest.raises(OntologyRoleError, match="no HP: terms"):
            loader(tmp_path).load(path, expect="hpo")

    def test_a_file_declaring_nothing_but_carrying_the_right_terms_loads(self, tmp_path):
        """**Absent is not contradictory.** OBO files without an `ontology:`
        tag exist, and refusing them on the declaration ground would never
        reach the term check."""
        path = obo(tmp_path / "anything.obo", ontology=None, terms=("HP:1",))

        assert loader(tmp_path).load(path, expect="hpo").num_terms == 1

    def test_cross_namespace_terms_do_not_make_a_file_wrong(self, tmp_path):
        """**Acceptance 18.** A legitimate ontology cross-references other
        namespaces — `add_ontology` skips those by design — so "every term
        carries one prefix" would refuse genuine MONDO."""
        path = obo(tmp_path / "mondo.obo", ontology="mondo",
                   terms=("MONDO:1", "HP:9", "GO:5", "MONDO:2"))

        assert loader(tmp_path).load(path, expect="mondo").num_terms == 4

    @pytest.mark.parametrize("spelling", ["hp", "hpo", "HP", "hp.obo"])
    def test_hp_and_hpo_are_the_same_slot(self, tmp_path, spelling):
        """**Acceptance 19.** The file the real PURL serves declares
        `ontology: hp.obo`, so a check that compared strings would refuse the
        genuine artifact."""
        path = obo(tmp_path / "hpo.obo", ontology=spelling, terms=("HP:1",))

        assert loader(tmp_path).load(path, expect="hpo").num_terms == 1

    def test_the_check_runs_before_load_returns(self, tmp_path):
        """A check a caller has to remember is a check that is missing wherever
        somebody forgot. `load(..., expect=...)` raises rather than returning
        something a caller must then inspect."""
        path = obo(tmp_path / "hpo.obo", ontology="mondo", terms=("MONDO:1",))

        with pytest.raises(OntologyRoleError):
            result = loader(tmp_path).load(path, expect="hpo")
            assert result is None, "unreachable"

    def test_an_ontology_with_no_known_prefix_is_not_refused_for_that(self, tmp_path):
        """Refusing here would be refusing the unknown rather than a mismatch.
        The declaration check is all this project can say about an ontology it
        has no prefix for."""
        class Fake:
            name = "chebi"

            def get_all_terms(self, include_obsolete=False):
                return ["CHEBI:1"]

        assert term_prefix("chebi") is None
        check_ontology_role(Fake(), "chebi")


class TestOneHomeForThePrefixes:

    def test_the_builder_reads_the_same_table(self):
        """**Two copies is how two tables come to disagree about what `HP:`
        means**, and the disagreement would be invisible: the builder silently
        skips terms, and the role check silently accepts a file.

        Scoped to the *selection* table. `add_ontology` also maps a term's
        prefix to a `DataSource`, which is a different association — it tags
        foreign terms too, so a `HP:` term inside MONDO is sourced to HPO — and
        folding the two together is not what this phase was asked for.
        """
        import inspect

        from src.kg.builder import KnowledgeGraphBuilder

        source = inspect.getsource(KnowledgeGraphBuilder.add_ontology)
        selection = source.split("EXPECTED_PREFIXES")[1].split("}")[0]

        for slot in ("mondo", "hpo", "go", "mp"):
            assert f'term_prefix("{slot}")' in selection
        for literal in ('"MONDO:"', '"HP:"', '"GO:"', '"MP:"'):
            assert literal not in selection, "the selection table has its own copy again"

    def test_the_association_is_what_the_builder_selects_by(self):
        """The check must measure against the prefix `add_ontology` will use,
        or "passes the check, then builds zero nodes" becomes reachable again."""
        from src.core.types import NodeType
        from src.kg.builder import KnowledgeGraphBuilder

        builder = KnowledgeGraphBuilder()
        for slot, node_type in (("mondo", NodeType.DISEASE), ("hpo", NodeType.PHENOTYPE)):
            prefix = term_prefix(slot)
            assert prefix is not None
            ontology = _stub(prefix, declares=slot)
            check_ontology_role(ontology, slot)
            assert builder.add_ontology(ontology, node_type) > 0, slot

    def test_every_named_ontology_has_a_prefix(self):
        for name in ("mondo", "hpo", "go", "mp"):
            assert term_prefix(name) == ONTOLOGY_TERM_PREFIXES[name]


def _stub(prefix: str, declares: str = "stub"):
    """The smallest object both the role check and `add_ontology` accept."""
    class Stub:
        name = declares

        def get_all_terms(self, include_obsolete=False):
            return [f"{prefix}0000001"]

        def get_term(self, term_id):
            return {"id": term_id, "name": "t", "definition": None,
                    "synonyms": [], "xrefs": [], "is_obsolete": False}

        def get_parents(self, term_id):
            return set()

        def to_edges(self):
            return []

    return Stub()
