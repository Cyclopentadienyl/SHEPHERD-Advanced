"""Choosing an ontology file, and refusing to choose when there is no basis.

`PLAN_ONTOLOGY_PHASE2.md` §3.1, §3.1.1, §3.1.2 and §3.2; acceptance 1-13.

**The case that carries the most weight is not a refusal.** An explicit path has
to keep working *while other valid versions sit on disk* — otherwise the rule
below would have quietly turned "we do not guess" into "you may only keep one
release", which is the opposite of what multi-version coexistence is for.

The second is that enumeration issues no network request. A term count normally
means a parse, and a parse at pronto's default resolves `import:` lines over the
network — so a resolver built the obvious way reaches the internet before anyone
has chosen a file.

Module: tests/unit/test_ontology_resolver.py
"""
from __future__ import annotations

import hashlib
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ontology.resolver import (  # noqa: E402
    AmbiguousOntologyError,
    NoOntologyCandidateError,
    OntologyResolutionError,
    canonical_ontology_name,
    enumerate_candidates,
    scan_identity,
    select_ontology_file,
)


def obo(path: Path, *, ontology="mondo", version="releases/2026-09-01",
        terms=("MONDO:0000001",), imports=()) -> Path:
    """An OBO file the way a real one is shaped: header stanza, then terms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["format-version: 1.2"]
    if version is not None:
        lines.append(f"data-version: {version}")
    if ontology is not None:
        lines.append(f"ontology: {ontology}")
    lines.extend(f"import: {item}" for item in imports)
    for term in terms:
        lines += ["", "[Term]", f"id: {term}", f"name: term {term}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def owl(path: Path, *, version="http://purl.obolibrary.org/obo/mondo/releases/2026-09-01/mondo.owl",
        iri="http://purl.obolibrary.org/obo/mondo.owl", classes=1, imports=()) -> Path:
    """Real RDF/XML, namespaces declared.

    **The first version of this helper emitted `<rdf:RDF>` with no `xmlns`
    declarations at all** — undeclared prefixes, so not well-formed XML and not
    something any loader would open. It satisfied a scanner matching literal
    text, which is how the scanner's namespace blindness went unnoticed. A
    fixture that no real reader accepts cannot prove a reader accepts it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f'  <owl:Ontology rdf:about="{iri}">']
    if version:
        body.append(f'    <owl:versionIRI rdf:resource="{version}"/>')
    body.extend(f'    <owl:imports rdf:resource="{item}"/>' for item in imports)
    body.append("  </owl:Ontology>")
    body += [
        f'  <owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_{i:07d}"/>'
        for i in range(classes)
    ]
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
        '         xmlns:owl="http://www.w3.org/2002/07/owl#">\n'
        + "\n".join(body)
        + "\n</rdf:RDF>\n"
    )
    return path


class TestIdentityWithoutAParser:
    """§3.2 — every field from one streaming pass, no pronto, no network."""

    def test_an_obo_file_reports_what_it_declares(self, tmp_path):
        path = obo(tmp_path / "mondo.obo", terms=("MONDO:1", "MONDO:2", "MONDO:3"))
        identity = scan_identity(path)

        assert identity["data_version"] == "releases/2026-09-01"
        assert identity["declared_ontology"] == "mondo"
        assert identity["term_count"] == 3
        assert identity["declared_imports"] == ()
        assert identity["size_bytes"] == path.stat().st_size

    def test_the_digest_is_of_the_bytes_on_disk(self, tmp_path):
        """Computed through a different entry point than the module uses, so a
        digest of the wrong thing — or of nothing — cannot pass."""
        path = obo(tmp_path / "mondo.obo")

        assert scan_identity(path)["digest"] == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_a_file_declaring_no_version_reports_none_rather_than_inventing(self, tmp_path):
        path = obo(tmp_path / "mondo.obo", version=None)

        assert scan_identity(path)["data_version"] is None

    def test_declared_imports_are_carried_before_anyone_chooses(self, tmp_path):
        """So a listing can say this file will be refused at load, rather than
        letting an operator pick it and meet the refusal afterwards."""
        path = obo(tmp_path / "mondo.obo", imports=("http://example.invalid/x.obo",))

        assert scan_identity(path)["declared_imports"] == ("http://example.invalid/x.obo",)

    def test_an_owl_file_reports_its_own_shape_and_says_which_it_counted(self, tmp_path):
        path = owl(tmp_path / "mondo.owl", classes=4, imports=("http://example.invalid/y.owl",))
        identity = scan_identity(path)

        assert identity["term_count"] == 4
        assert identity["term_count_basis"] == "owl:Class declarations"
        assert identity["declared_imports"] == ("http://example.invalid/y.owl",)
        assert "2026-09-01" in identity["data_version"]

    def test_the_two_counts_do_not_share_one_name(self, tmp_path):
        """`term_count` means different things in the two formats and the basis
        is recorded, so a report cannot compare them as though it did not."""
        a = scan_identity(obo(tmp_path / "mondo.obo"))
        b = scan_identity(owl(tmp_path / "hpo.owl"))

        assert a["term_count_basis"] != b["term_count_basis"]

    def test_a_directory_is_not_a_candidate_file(self, tmp_path):
        (tmp_path / "mondo.obo").mkdir()

        with pytest.raises(OntologyResolutionError, match="not a regular file"):
            scan_identity(tmp_path / "mondo.obo")


class TestEnumerationTouchesNothingButTheFilesystem:

    def test_it_issues_no_network_request(self, tmp_path, monkeypatch):
        """**Acceptance 13.** A term count normally means a parse, and pronto's
        default resolves imports over the network. Every socket entry point is
        replaced with a detonator, including for a file that declares one."""
        def explode(*a, **k):
            raise AssertionError("enumeration reached the network")

        for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
            monkeypatch.setattr(socket, name, explode, raising=False)

        obo(tmp_path / "roots" / "mondo.obo",
            imports=("http://purl.obolibrary.org/obo/x.obo",))
        obo(tmp_path / "roots" / "hpo.obo", ontology="hpo", terms=("HP:1",))

        found = enumerate_candidates([tmp_path / "roots"])

        assert {item.ontology for item in found} == {"mondo", "hpo"}

    def test_it_creates_no_directories(self, tmp_path):
        """**Acceptance 12.** `OntologyLoader.__init__` mkdirs its cache, so a
        resolver written beside it is one line away from doing the same."""
        absent = tmp_path / "not_here"

        assert enumerate_candidates([absent]) == []
        assert not absent.exists()

    def test_a_root_that_is_a_file_is_skipped_rather_than_raising(self, tmp_path):
        (tmp_path / "a_file").write_text("x")

        assert enumerate_candidates([tmp_path / "a_file"]) == []

    def test_an_unreadable_file_does_not_stop_the_others(self, tmp_path):
        root = tmp_path / "roots"
        obo(root / "mondo.obo")
        (root / "hpo.obo").mkdir()

        found = enumerate_candidates(root_list := [root])

        assert [item.path.name for item in found] == ["mondo.obo"]
        assert root_list  # the call did not consume the roots

    def test_filtering_is_by_what_the_file_declares_not_its_name(self, tmp_path):
        """A file named `hpo.obo` that declares `mondo` is not an HPO candidate.
        Listing it as one would turn §3.4's clear mismatch into an ambiguity."""
        root = tmp_path / "roots"
        obo(root / "hpo.obo", ontology="mondo")
        obo(root / "real_hpo.obo", ontology="hpo", terms=("HP:1",))

        found = enumerate_candidates([root], ontology="hpo")

        assert [item.path.name for item in found] == ["real_hpo.obo"]

    def test_a_file_declaring_nothing_falls_back_to_its_filename(self, tmp_path):
        root = tmp_path / "roots"
        obo(root / "mondo.obo", ontology=None)

        assert len(enumerate_candidates([root], ontology="mondo")) == 1


class TestTheAliases:

    @pytest.mark.parametrize("spelling", ["hp", "hpo", "HP", "hp.obo", "human_phenotype"])
    def test_hp_and_hpo_are_one_ontology(self, spelling):
        """**Acceptance 19's half that lives here.** A resolver treating them as
        two would refuse a directory holding one file under each name."""
        assert canonical_ontology_name(spelling) == "hpo"

    def test_a_directory_holding_hp_and_hpo_is_one_ambiguity_not_two_ontologies(self, tmp_path):
        root = tmp_path / "roots"
        obo(root / "hp.obo", ontology="hp", terms=("HP:1",))
        obo(root / "hpo.obo", ontology="hpo", terms=("HP:2",))

        with pytest.raises(AmbiguousOntologyError) as caught:
            select_ontology_file("hpo", roots=[root])

        assert len(caught.value.candidates) == 2

    @pytest.mark.parametrize("value", [None, "", "   ", 5, b"hpo"])
    def test_a_value_that_is_not_a_name_is_not_one(self, value):
        assert canonical_ontology_name(value) is None


class TestSelection:
    """§3.1's precedence table, which is the whole contract."""

    def test_an_explicit_path_is_used(self, tmp_path):
        path = obo(tmp_path / "somewhere" / "my_mondo.obo")

        chosen = select_ontology_file("mondo", roots=[tmp_path], explicit_path=path)

        assert chosen.path == path

    def test_an_explicit_path_wins_while_other_versions_sit_on_disk(self, tmp_path):
        """**Acceptance 2, and the case the first draft of the plan omitted.**
        The rule is a default about ambiguity, not a limit on coexistence: an
        operator keeps every release they have and names the one they mean. No
        file is deleted and no code is changed.
        """
        root = tmp_path / "roots"
        obo(root / "mondo.obo", version="releases/2026-01-01")
        obo(root / "mondo_new.obo", version="releases/2026-09-01")
        wanted = obo(tmp_path / "elsewhere" / "mondo.obo", version="releases/2025-05-05")

        chosen = select_ontology_file("mondo", roots=[root], explicit_path=wanted)

        assert chosen.path == wanted
        assert chosen.data_version == "releases/2025-05-05"
        assert chosen.digest == hashlib.sha256(wanted.read_bytes()).hexdigest(), (
            "provenance would record a rival's digest"
        )

    def test_an_explicit_path_that_does_not_exist_refuses(self, tmp_path):
        """**Acceptance 1.** It must not fall through to the cache convention —
        a build that quietly used another file is the failure being removed."""
        obo(tmp_path / "roots" / "mondo.obo")

        with pytest.raises(OntologyResolutionError, match="does not exist"):
            select_ontology_file("mondo", roots=[tmp_path / "roots"],
                                 explicit_path=tmp_path / "gone.obo")

    def test_an_explicit_path_that_is_a_directory_refuses(self, tmp_path):
        (tmp_path / "mondo.obo").mkdir()

        with pytest.raises(OntologyResolutionError, match="not a regular file"):
            select_ontology_file("mondo", explicit_path=tmp_path / "mondo.obo")

    def test_exactly_one_candidate_is_taken(self, tmp_path):
        """**Acceptance 5.** Without this the refusals above would be satisfied
        by a resolver that rejects everything."""
        root = tmp_path / "roots"
        path = obo(root / "mondo.obo")

        assert select_ontology_file("mondo", roots=[root]).path == path

    def test_more_than_one_candidate_refuses_and_names_them(self, tmp_path):
        """**Acceptance 3.** The message has to carry what an operator needs to
        take the decision, and has to say how."""
        root = tmp_path / "roots"
        obo(root / "mondo.obo", version="releases/2026-01-01")
        obo(root / "mondo_old.obo", version="releases/2025-01-01")

        with pytest.raises(AmbiguousOntologyError) as caught:
            select_ontology_file("mondo", roots=[root])

        message = str(caught.value)
        assert "releases/2026-01-01" in message and "releases/2025-01-01" in message
        assert "--mondo-path" in message
        assert len(caught.value.candidates) == 2

    def test_two_roots_each_holding_one_refuse(self, tmp_path):
        """**Acceptance 3's cross-root half**, and the reason root order is not
        precedence: whichever came first would otherwise win silently."""
        obo(tmp_path / "a" / "mondo.obo", version="releases/2026-01-01")
        obo(tmp_path / "b" / "mondo.obo", version="releases/2025-01-01")

        with pytest.raises(AmbiguousOntologyError):
            select_ontology_file("mondo", roots=[tmp_path / "a", tmp_path / "b"])

    def test_root_order_does_not_decide(self, tmp_path):
        """The same two roots in the other order refuse identically. A resolver
        that picked the first would pass the test above and fail here."""
        obo(tmp_path / "a" / "mondo.obo", version="releases/2026-01-01")
        obo(tmp_path / "b" / "mondo.obo", version="releases/2025-01-01")

        for order in ([tmp_path / "a", tmp_path / "b"], [tmp_path / "b", tmp_path / "a"]):
            with pytest.raises(AmbiguousOntologyError):
                select_ontology_file("mondo", roots=order)

    def test_the_newest_data_version_does_not_win_either(self, tmp_path):
        """Named separately because it is the rule most likely to be added as a
        convenience. Ordering by a string a file declares about itself is a
        guess wearing a comparison."""
        root = tmp_path / "roots"
        obo(root / "a.obo", version="releases/2030-01-01")
        obo(root / "b.obo", version="releases/1999-01-01")

        with pytest.raises(AmbiguousOntologyError):
            select_ontology_file("mondo", roots=[root])

    def test_obo_is_not_preferred_over_owl(self, tmp_path):
        """**Acceptance 8.** Today `_load_known_ontology` takes the OBO
        silently. The two are not guaranteed to be one release in two
        encodings — the download fallback writes whichever succeeded, whenever
        it ran."""
        root = tmp_path / "roots"
        obo(root / "mondo.obo", version="releases/2026-09-01")
        owl(root / "mondo.owl")

        with pytest.raises(AmbiguousOntologyError) as caught:
            select_ontology_file("mondo", roots=[root])

        assert {item.path.suffix for item in caught.value.candidates} == {".obo", ".owl"}

    def test_that_refusal_shows_both_versions_so_the_choice_is_easy(self, tmp_path):
        """When they are the same release in two encodings the operator sees it
        at once. The refusal is meant to inform, not to obstruct."""
        root = tmp_path / "roots"
        obo(root / "mondo.obo", version="releases/2026-09-01")
        owl(root / "mondo.owl")

        with pytest.raises(AmbiguousOntologyError) as caught:
            select_ontology_file("mondo", roots=[root])

        assert str(caught.value).count("2026-09-01") >= 2

    def test_naming_a_path_resolves_that_ambiguity(self, tmp_path):
        """**Acceptance 10.** The documented migration has to actually work."""
        root = tmp_path / "roots"
        wanted = obo(root / "mondo.obo")
        owl(root / "mondo.owl")

        assert select_ontology_file("mondo", roots=[root], explicit_path=wanted).path == wanted

    def test_no_candidate_is_its_own_outcome(self, tmp_path):
        """Not a refusal of the workspace: the caller may download, and it
        cannot decide that if this arrives as the same exception as an
        ambiguous directory."""
        with pytest.raises(NoOntologyCandidateError) as caught:
            select_ontology_file("mondo", roots=[tmp_path])

        assert caught.value.roots == (tmp_path,)
        assert not isinstance(caught.value, AmbiguousOntologyError)

    def test_no_roots_at_all_is_also_that_outcome(self):
        with pytest.raises(NoOntologyCandidateError):
            select_ontology_file("mondo", roots=[])

    def test_another_ontologys_file_is_not_a_candidate(self, tmp_path):
        root = tmp_path / "roots"
        obo(root / "hpo.obo", ontology="hpo", terms=("HP:1",))

        with pytest.raises(NoOntologyCandidateError):
            select_ontology_file("mondo", roots=[root])

    def test_the_same_file_reached_through_two_roots_is_one_candidate(self, tmp_path):
        """A root and a symlink to it, or a root nested in another, is a
        configuration mistake — not a reason to refuse a build."""
        root = tmp_path / "roots"
        path = obo(root / "mondo.obo")
        link = tmp_path / "alias"
        try:
            link.symlink_to(root, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("this platform does not allow the symlink")

        assert select_ontology_file("mondo", roots=[root, link]).path == path


class TestOwlIsReadAsXmlNotAsText:
    """The scanner defeating the very refusal it feeds.

    The first version matched the literal strings `owl:Class` and `rdf:about`
    line by line. A prefix is chosen by the document, and an attribute may sit
    on a different line from its element — so a real, loadable OWL file
    reported no ontology and no terms, was filtered out of its own candidate
    list, and a directory holding it beside an `.obo` resolved to **one**
    candidate and was taken silently. Losing a display field would be a
    nuisance; losing the ambiguity is the rule not working.
    """

    RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    OWL = "http://www.w3.org/2002/07/owl#"

    def _swapped_prefixes(self, path: Path) -> Path:
        path.write_text(
            '<?xml version="1.0"?>\n'
            f'<o:RDF xmlns:o="{self.RDF}" xmlns:w="{self.OWL}">\n'
            '<w:Ontology o:about="http://purl.obolibrary.org/obo/mondo.owl">\n'
            '  <w:versionIRI o:resource="http://purl.obolibrary.org/obo/mondo/'
            'releases/2026-09-01/mondo.owl"/>\n'
            '</w:Ontology>\n'
            '<w:Class o:about="http://purl.obolibrary.org/obo/MONDO_0000001"/>\n'
            '</o:RDF>\n'
        )
        return path

    def _attribute_on_the_next_line(self, path: Path) -> Path:
        path.write_text(
            f'<rdf:RDF xmlns:rdf="{self.RDF}" xmlns:owl="{self.OWL}">\n'
            '<owl:Ontology\n'
            '    rdf:about="http://purl.obolibrary.org/obo/mondo.owl"/>\n'
            '<owl:Class\n'
            '    rdf:about="http://purl.obolibrary.org/obo/MONDO_0000002"/>\n'
            '<owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_0000003"/>\n'
            '</rdf:RDF>\n'
        )
        return path

    def test_a_document_choosing_its_own_prefixes_is_identified(self, tmp_path):
        identity = scan_identity(self._swapped_prefixes(tmp_path / "a.owl"))

        assert canonical_ontology_name(identity["declared_ontology"]) == "mondo"
        assert identity["term_count"] == 1
        assert "2026-09-01" in (identity["data_version"] or "")

    def test_an_attribute_on_the_next_line_is_identified(self, tmp_path):
        identity = scan_identity(self._attribute_on_the_next_line(tmp_path / "b.owl"))

        assert canonical_ontology_name(identity["declared_ontology"]) == "mondo"
        assert identity["term_count"] == 2

    def test_an_import_is_found_whatever_the_prefix(self, tmp_path):
        path = tmp_path / "c.owl"
        path.write_text(
            f'<r:RDF xmlns:r="{self.RDF}" xmlns:w="{self.OWL}">\n'
            '<w:Ontology r:about="http://purl.obolibrary.org/obo/hp.owl">\n'
            '<w:imports\n    r:resource="http://purl.obolibrary.org/obo/pato.owl"/>\n'
            '</w:Ontology></r:RDF>\n'
        )

        assert scan_identity(path)["declared_imports"] == (
            "http://purl.obolibrary.org/obo/pato.owl",
        )

    @pytest.mark.parametrize("shape", ["_swapped_prefixes", "_attribute_on_the_next_line"])
    def test_such_a_file_is_a_candidate_and_the_ambiguity_survives(self, tmp_path, shape):
        """The case that matters: the refusal, not the field."""
        root = tmp_path / "roots"
        obo(root / "mondo.obo", version="releases/2026-09-01")
        getattr(self, shape)(root / "release-2026.owl")

        with pytest.raises(AmbiguousOntologyError) as caught:
            select_ontology_file("mondo", roots=[root])

        assert {item.path.suffix for item in caught.value.candidates} == {".obo", ".owl"}

    def test_a_file_that_is_not_well_formed_xml_is_reported(self, tmp_path):
        path = tmp_path / "broken.owl"
        path.write_text('<rdf:RDF xmlns:rdf="%s"><owl:Ontology' % self.RDF)

        with pytest.raises(OntologyResolutionError, match="well-formed"):
            scan_identity(path)

    def test_and_it_is_skipped_rather_than_failing_the_listing(self, tmp_path):
        """One unusable file does not stop the others being described."""
        root = tmp_path / "roots"
        obo(root / "mondo.obo")
        (root / "broken.owl").write_text("<rdf:RDF")

        assert [item.path.name for item in enumerate_candidates([root])] == ["mondo.obo"]

    def test_the_xml_reader_issues_no_request_for_an_import(self, tmp_path, monkeypatch):
        """`ElementTree` has no notion of `owl:imports`, and this pins it: the
        parser that replaced the regex must not be one that resolves."""
        def explode(*a, **k):
            raise AssertionError("the OWL scan reached the network")

        for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
            monkeypatch.setattr(socket, name, explode, raising=False)

        path = tmp_path / "d.owl"
        path.write_text(
            f'<rdf:RDF xmlns:rdf="{self.RDF}" xmlns:owl="{self.OWL}">\n'
            '<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/mondo.owl">\n'
            '<owl:imports rdf:resource="http://purl.obolibrary.org/obo/pato.owl"/>\n'
            '</owl:Ontology></rdf:RDF>\n'
        )

        assert scan_identity(path)["declared_imports"]


class TestTheOwlScanHoldsNoTree:
    """**Incremental feeding is not the same as bounded memory.**

    The scanner that replaced the regex used `XMLPullParser`, whose default tree
    builder keeps every element attached to its parent until the document ends;
    reading its events empties a queue and frees nothing. Measured on generated
    RDF/XML with a label and a subClassOf per class, the peak was about five
    times the file — 36 MB in, 182 MB held — while a comment above the code
    said "bounded". Enumeration scans every candidate before it filters, so a
    large MONDO OWL was built into memory even when the build wanted HPO.

    The target parser keeps counts and no elements. These cases pin the shape of
    that, not a number from one machine: the peak must not grow with the file.
    """

    RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    OWL = "http://www.w3.org/2002/07/owl#"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"

    def _generate(self, path: Path, classes: int) -> Path:
        with open(path, "w") as out:
            out.write(
                f'<?xml version="1.0"?>\n<rdf:RDF xmlns:rdf="{self.RDF}" '
                f'xmlns:owl="{self.OWL}" xmlns:rdfs="{self.RDFS}">\n'
                '<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/mondo.owl"/>\n'
            )
            for i in range(classes):
                out.write(
                    f'<owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_{i:07d}">\n'
                    f"  <rdfs:label>disease number {i} with a reasonably long label</rdfs:label>\n"
                    f'  <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/'
                    f'MONDO_{max(i - 1, 0):07d}"/>\n'
                    "</owl:Class>\n"
                )
            out.write("</rdf:RDF>\n")
        return path

    @staticmethod
    def _peak(path: Path):
        import tracemalloc

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            identity = scan_identity(path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return identity, peak

    def test_the_peak_does_not_grow_with_the_file(self, tmp_path):
        small_path = self._generate(tmp_path / "small.owl", 2_000)
        large_path = self._generate(tmp_path / "large.owl", 20_000)

        small, small_peak = self._peak(small_path)
        large, large_peak = self._peak(large_path)

        assert small["term_count"] == 2_000 and large["term_count"] == 20_000
        # Ten times the file must not mean ten times the memory. The tree
        # builder measured ~2.4 MB here against ~24 MB; the margin below is for
        # interpreter noise, not for a tree that is retained after all.
        assert large_peak < small_peak * 2 + 64 * 1024, (
            f"peak grew with the file: {small_peak:,} -> {large_peak:,} bytes"
        )
        assert large_peak < large_path.stat().st_size // 4, (
            f"peak {large_peak:,} bytes is a real fraction of a "
            f"{large_path.stat().st_size:,}-byte file; something is being kept"
        )

    def test_the_digest_is_still_of_the_bytes_the_parser_saw(self, tmp_path):
        """One pass for both answers, so the identity and the digest describe
        the same bytes."""
        path = self._generate(tmp_path / "m.owl", 50)

        assert scan_identity(path)["digest"] == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_an_unterminated_document_is_still_refused(self, tmp_path):
        """Dropping the tree must not drop the end-of-document check: a file
        cut off mid-element is not a smaller release."""
        path = self._generate(tmp_path / "m.owl", 10)
        path.write_bytes(path.read_bytes()[:-20])

        with pytest.raises(OntologyResolutionError, match="well-formed"):
            scan_identity(path)
