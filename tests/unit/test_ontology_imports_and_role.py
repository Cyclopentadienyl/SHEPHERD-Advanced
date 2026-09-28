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

import os
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


class TestTheLoaderAndTheListingAgreeOnImports:
    """pronto records only `owl:imports` directly under the first
    `owl:Ontology`; the listing records them wherever RDF/XML puts them. With
    the loader refusing on pronto's set alone, a file listed as "will be
    refused" loaded with its import silently dropped — the suppression §3.3
    exists to prevent. The loader now refuses on the union."""

    RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    OWL = "http://www.w3.org/2002/07/owl#"

    def _owl(self, path: Path, body: str) -> Path:
        path.write_text(
            f'<?xml version="1.0"?>\n<rdf:RDF xmlns:rdf="{self.RDF}" xmlns:owl="{self.OWL}">\n'
            + body + "\n</rdf:RDF>\n"
        )
        return path

    def test_an_import_in_an_rdf_description_is_refused_at_load(self, tmp_path):
        path = self._owl(tmp_path / "hp.owl", """
<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/hp.owl"/>
<rdf:Description rdf:about="http://purl.obolibrary.org/obo/hp.owl">
  <owl:imports rdf:resource="http://purl.obolibrary.org/obo/hp/imports/uberon_import.owl"/>
</rdf:Description>
<owl:Class rdf:about="http://purl.obolibrary.org/obo/HP_0000001"/>""")

        with pytest.raises(OntologyImportError, match="uberon_import"):
            loader(tmp_path).load(path)

    def test_an_import_written_as_a_nested_ontology_is_refused_and_named(self, tmp_path):
        """pronto records `None` for this shape, so the refusal used to list
        "None" while the listing reported no import at all."""
        path = self._owl(tmp_path / "hp.owl", """
<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/hp.owl">
  <owl:imports><owl:Ontology rdf:about="http://purl.obolibrary.org/obo/pato.owl"/></owl:imports>
</owl:Ontology>
<owl:Class rdf:about="http://purl.obolibrary.org/obo/HP_0000001"/>""")

        with pytest.raises(OntologyImportError) as caught:
            loader(tmp_path).load(path)

        assert "pato.owl" in str(caught.value)
        assert "None" not in str(caught.value)

    def test_an_import_with_no_target_is_still_refused(self, tmp_path):
        """`<owl:imports/>` names nothing, and pronto records `{None}` for it —
        measured. The listing has no target to show, so this refusal rests on
        pronto's `None` alone; dropping it loads the file as self-contained."""
        path = self._owl(tmp_path / "hp.owl", """
<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/hp.owl">
  <owl:imports/>
</owl:Ontology>
<owl:Class rdf:about="http://purl.obolibrary.org/obo/HP_0000001"/>""")

        with pytest.raises(OntologyImportError, match="no target"):
            loader(tmp_path).load(path)

    @pytest.mark.parametrize("declaration", [
        "<owl:imports/>",
        '<owl:imports rdf:resource=""/>',
        '<owl:imports rdf:nodeID="dependency"/>',
    ], ids=["empty", "empty-resource", "blank-node"])
    @pytest.mark.parametrize("where", ["description", "ontology"])
    def test_a_targetless_import_is_refused_wherever_it_is_written(self, tmp_path, declaration, where):
        """The reviewer's three reproductions, and the combination the first
        no-target test missed: pronto records `None` only for a declaration
        directly under the first `owl:Ontology`, so one inside an
        `rdf:Description` left no trace in pronto *or* the listing, and the
        file loaded as self-contained. A declaration is a declaration whether
        or not a target can be read from it."""
        from src.ontology.resolver import IMPORT_WITHOUT_TARGET, declared_imports

        iri = "http://purl.obolibrary.org/obo/mondo.owl"
        if where == "description":
            body = (f'<owl:Ontology rdf:about="{iri}"/>\n'
                    f'<rdf:Description rdf:about="{iri}">{declaration}</rdf:Description>')
        else:
            body = f'<owl:Ontology rdf:about="{iri}">{declaration}</owl:Ontology>'
        path = self._owl(tmp_path / "mondo.owl",
                         body + '\n<owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_0000001"/>')

        assert declared_imports(path) == (IMPORT_WITHOUT_TARGET,)
        with pytest.raises(OntologyImportError, match="no target"):
            loader(tmp_path).load(path, expect="mondo")

    def test_an_obo_import_with_no_value_is_listed_as_one(self, tmp_path):
        """fastobo refuses to parse it, so it never loads; the listing still
        has to show it rather than call the file self-contained."""
        from src.ontology.resolver import IMPORT_WITHOUT_TARGET, declared_imports, scan_identity

        path = tmp_path / "mondo.obo"
        path.write_text("format-version: 1.2\nontology: mondo\nimport: \n\n"
                        "[Term]\nid: MONDO:0000001\nname: d\n")

        assert declared_imports(path) == (IMPORT_WITHOUT_TARGET,)
        assert scan_identity(path)["declared_imports"] == (IMPORT_WITHOUT_TARGET,)

    def test_the_obo_import_check_reads_only_the_header(self, tmp_path, monkeypatch):
        """So agreement costs a few kilobytes on a large MONDO, not a pass —
        and an `import:` line after the first stanza, which OBO does not treat
        as a header tag, is not reported as one."""
        from src.ontology import resolver

        path = obo(tmp_path / "mondo.obo", ontology="mondo", terms=("MONDO:1",) * 3,
                   imports=("http://x.invalid/a.obo",))
        with open(path, "a") as handle:
            handle.write("\n[Term]\nid: MONDO:2\nimport: http://x.invalid/after-the-header.obo\n")
        monkeypatch.setattr(resolver, "scan_identity",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("full scan")))

        assert resolver.declared_imports(path) == ("http://x.invalid/a.obo",)


#: Windows refuses to replace a file another handle holds open (Python opens
#: without `FILE_SHARE_DELETE`); POSIX replaces the name and leaves the open
#: handle on the old file. Measured by the reviewer on CPython 3.13.9: the
#: replacement below raised `PermissionError: [WinError 5]`.
_REPLACING_AN_OPEN_FILE_IS_REFUSED = os.name == "nt"


class TestTheDigestIsOfWhatWasParsed:

    def test_a_replacement_during_the_load_changes_none_of_the_three_readings(
        self, tmp_path, monkeypatch
    ):
        """Digest, parse and import scan are taken through one open handle. The
        path is replaced — by a file of another release that also declares an
        import — after the file is opened and before pronto parses it: what
        comes back is still entirely the original.

        **Two platforms, one guarantee, reached two ways.** On POSIX the
        replacement happens and the open handle keeps reading the old file; on
        Windows the operating system refuses the replacement while the file is
        open. Either way the three readings are of one file. Only that one
        refusal, on that one platform, is accepted — anywhere else a
        `PermissionError` fails the test — and each branch asserts what its
        platform actually did, so neither can pass by skipping the other's
        checks."""
        import hashlib
        import types

        import pronto

        import src.ontology.loader as loader_module

        path = obo(tmp_path / "mondo.obo", ontology="mondo", terms=("MONDO:0000001",))
        original = path.read_bytes()
        replacement = obo(tmp_path / "next.obo", ontology="mondo", terms=("MONDO:0000002",),
                          imports=("http://x.invalid/late.obo",))
        attempt = {}

        def parse_after_the_path_moved(handle, **kwargs):
            try:
                os.replace(replacement, path)
                attempt["replaced"] = True
            except PermissionError:
                if not _REPLACING_AN_OPEN_FILE_IS_REFUSED:
                    raise
                attempt["replaced"] = False
            return pronto.Ontology(handle, **kwargs)

        monkeypatch.setattr(loader_module, "pronto",
                            types.SimpleNamespace(Ontology=parse_after_the_path_moved))

        loaded = loader(tmp_path).load(path, expect="mondo")

        if attempt["replaced"]:
            assert path.read_bytes() != original, "the replacement did not reach the path"
        else:
            assert path.read_bytes() == original, "the refused replacement changed the file"
            assert replacement.exists(), "the refused replacement consumed its source"
        if not _REPLACING_AN_OPEN_FILE_IS_REFUSED:
            assert attempt["replaced"], "POSIX must exercise the replacement, not skip it"
        assert loaded.source_digest == hashlib.sha256(original).hexdigest()
        assert loaded.has_term("MONDO:0000001") and not loaded.has_term("MONDO:0000002")


class TestAReleaseProductIsItsOntology:

    @pytest.mark.parametrize("declared", ["mondo/mondo-base", "mondo-simple"])
    def test_an_explicitly_named_variant_passes_the_role_check(self, tmp_path, declared):
        """It was refused by a message saying it would build zero nodes — while
        it carried the right terms. The term test still guards the content."""
        path = obo(tmp_path / "mondo-base.obo", ontology=declared, terms=("MONDO:0000001",))

        assert loader(tmp_path).load(path, expect="mondo").num_terms == 1

    def test_a_variant_of_another_ontology_is_still_refused(self, tmp_path):
        path = obo(tmp_path / "x.obo", ontology="hp/hp-base", terms=("HP:1",))

        with pytest.raises(OntologyRoleError):
            loader(tmp_path).load(path, expect="mondo")
