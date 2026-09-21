"""The build's precedence table, and the behaviour change it is honest about.

`PLAN_ONTOLOGY_PHASE2.md` §3.1 and §3.1.2; acceptance 6, 7, 9, 10, 20, 30.

**Placing the cache last does not preserve its old behaviour, and this file is
where that is admitted rather than claimed away.** `_load_known_ontology`
prefers `<name>.obo` over `<name>.owl` in one directory; under §3.1 both are
candidates and the build refuses — and a cache holding both is exactly what the
OBO-to-OWL download fallback produces. The migration is real, it is deliberate,
and the tests below cover both halves: the refusal, and the documented fix
working.

Module: tests/unit/test_ontology_selection_cli.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.ontology.loader import OntologyLoader  # noqa: E402
from src.ontology.roles import OntologyRoleError  # noqa: E402
from src.ontology.settings import load_ontology_settings as _genuine_settings  # noqa: E402

#: Captured before any fixture patches the module attribute. A test that reads
#: it back from the module afterwards gets the *patched* function and its own
#: config path is then silently ignored — which is how the cross-root case
#: below first passed for the wrong reason.


def build_module():
    """The real build script, loaded as the script it is."""
    spec = importlib.util.spec_from_file_location(
        "build_knowledge_graph", REPO / "scripts" / "build_knowledge_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obo(path: Path, *, ontology="mondo", version="releases/2026-09-01",
        terms=("MONDO:0000001",)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["format-version: 1.2", f"data-version: {version}"]
    if ontology:
        lines.append(f"ontology: {ontology}")
    for term in terms:
        lines += ["", "[Term]", f"id: {term}", f"name: term {term}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def owl(path: Path, *, version="http://purl.obolibrary.org/obo/mondo/2026-09-01/mondo.owl",
        iri="http://purl.obolibrary.org/obo/mondo.owl") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<rdf:RDF>\n"
        f'<owl:Ontology rdf:about="{iri}">\n'
        f'  <owl:versionIRI rdf:resource="{version}"/>\n'
        "</owl:Ontology>\n"
        '<owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_0000001"/>\n'
        "</rdf:RDF>\n"
    )
    return path


@pytest.fixture
def select(tmp_path, monkeypatch):
    """`_select_and_load` with the deployment configuration held empty.

    Without this the test would read the committed `configs/deployment.yaml`,
    which is a different input on a machine that has configured roots.
    """
    module = build_module()
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n")

    import src.ontology.settings as settings_module

    monkeypatch.setattr(
        settings_module, "load_ontology_settings",
        lambda config_path=None: _genuine_settings(empty),
    )

    def run(*, ontology="mondo", explicit_path=None, cache_dir=None,
            force_download=False, loader=None):
        return module._select_and_load(
            loader or OntologyLoader(cache_dir=tmp_path / "loader_cache"),
            ontology, explicit_path, cache_dir, force_download,
        )

    return run


class TestThePrecedenceTable:

    def test_an_explicit_path_is_loaded(self, tmp_path, select):
        path = obo(tmp_path / "somewhere" / "my_mondo.obo")

        assert select(explicit_path=path).source_path == path

    def test_an_explicit_path_wins_over_the_cache(self, tmp_path, select):
        """**Acceptance 2 at the build level.** The cache holds a different
        release and is not consulted."""
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2020-01-01")
        wanted = obo(tmp_path / "wanted.obo", version="releases/2026-09-01")

        loaded = select(explicit_path=wanted, cache_dir=cache)

        assert loaded.source_path == wanted
        assert loaded.declared_version == "releases/2026-09-01"

    def test_exactly_one_candidate_in_the_cache_still_works(self, tmp_path, select):
        """**Acceptance 7.** The common case, and the one an existing
        invocation is."""
        cache = tmp_path / "cache"
        path = obo(cache / "mondo.obo")

        assert select(cache_dir=cache).source_path == path

    def test_a_cache_holding_obo_and_owl_refuses(self, tmp_path, select):
        """**Acceptance 8 at the build level, and the behaviour change.** Today
        the OBO wins silently; a cache holding both is what the OBO-to-OWL
        download fallback produces."""
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2026-09-01")
        owl(cache / "mondo.owl")

        with pytest.raises(SystemExit) as caught:
            select(cache_dir=cache)

        message = str(caught.value)
        assert "mondo.obo" in message and "mondo.owl" in message
        assert "--mondo-path" in message, "the migration is not named"

    def test_naming_a_path_fixes_that(self, tmp_path, select):
        """**Acceptance 10.** The documented migration has to work, or the
        refusal above is a dead end rather than a redirection."""
        cache = tmp_path / "cache"
        wanted = obo(cache / "mondo.obo")
        owl(cache / "mondo.owl")

        assert select(explicit_path=wanted, cache_dir=cache).source_path == wanted

    def test_a_root_and_the_cache_each_holding_one_refuses(self, tmp_path, select,
                                                           monkeypatch):
        """**Acceptance 9.** The cache being last does not decide it, because
        the policy does not select by root order."""
        import src.ontology.settings as settings_module

        root = tmp_path / "configured"
        obo(root / "mondo.obo", version="releases/2026-09-01")
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2020-01-01")

        config = tmp_path / "with_root.yaml"
        config.write_text(f"paths:\n  ontology_roots:\n    - {root}\n")
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(config))
        assert _genuine_settings(config).roots == (root,), "the configured root did not take"

        with pytest.raises(SystemExit, match="--mondo-path"):
            select(cache_dir=cache)

    def test_the_loaders_default_cache_is_a_root_when_no_flag_is_given(self, tmp_path, select):
        """**Found by the existing suite, not by review.** The first version
        added the cache as a root only when `--ontology-cache-dir` was passed,
        which silently dropped `~/.shepherd/ontologies` — the default, and where
        every existing install's files already are. A build that used to open
        them would have found no candidate and re-downloaded."""
        default_cache = tmp_path / "default_cache"
        present = obo(default_cache / "mondo.obo")

        class Loader(OntologyLoader):
            def _download_ontology(self, name, force):
                raise AssertionError("re-downloaded a file that was already present")

        loaded = select(cache_dir=None, loader=Loader(cache_dir=default_cache))

        assert loaded.source_path == present

    def test_an_explicit_path_that_does_not_exist_refuses(self, tmp_path, select):
        """**Acceptance 1.** Not a hint: it does not fall through to a cache
        that happens to hold something."""
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo")

        with pytest.raises(ValueError, match="does not exist"):
            select(explicit_path=tmp_path / "gone.obo", cache_dir=cache)


class TestForceDownload:
    """Acceptance 6 — the flag that must not become decorative."""

    def test_with_an_explicit_path_it_refuses(self, tmp_path, select):
        path = obo(tmp_path / "mondo.obo")

        with pytest.raises(SystemExit, match="two different instructions"):
            select(explicit_path=path, force_download=True)

    def test_it_reaches_the_fetch_rather_than_the_resolver(self, tmp_path, select):
        """**The failure this is written against**: a flag that survives in the
        signature while the resolver returns before anything reads it. A cache
        holding a perfectly good file must not satisfy `--force-download`."""
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2020-01-01")
        fetched = obo(tmp_path / "fetched.obo", version="releases/2026-09-01")

        calls = []

        class Loader(OntologyLoader):
            def _download_ontology(self, name, force):
                calls.append((name, force))
                return fetched

        loaded = select(cache_dir=cache, force_download=True,
                        loader=Loader(cache_dir=cache))

        assert calls == [("mondo", True)], "the resolver short-circuited the flag"
        assert loaded.source_path == fetched

    def test_without_it_a_present_file_is_used_and_nothing_is_fetched(self, tmp_path, select):
        cache = tmp_path / "cache"
        present = obo(cache / "mondo.obo")

        class Loader(OntologyLoader):
            def _download_ontology(self, name, force):
                raise AssertionError("fetched despite a candidate being present")

        assert select(cache_dir=cache, loader=Loader(cache_dir=cache)).source_path == present


class TestTheRoleCheckReachesTheBuild:

    def test_a_wrong_slot_stops_the_build(self, tmp_path, select):
        """**Acceptance 17 through the entry point the operator uses.** The
        library check is only a gate if the build passes `expect`."""
        path = obo(tmp_path / "hpo.obo", ontology="mondo", terms=("MONDO:1",))

        with pytest.raises(OntologyRoleError):
            select(ontology="hpo", explicit_path=path)

    def test_it_does_not_fall_back_to_the_cache(self, tmp_path, select):
        """**Acceptance 20.** An explicitly named file that is wrong is an
        error to report, not a reason to open something else — and the cache
        here holds a file that would have loaded."""
        cache = tmp_path / "cache"
        obo(cache / "hpo.obo", ontology="hpo", terms=("HP:1",))
        wrong = obo(tmp_path / "wrong.obo", ontology="mondo", terms=("MONDO:1",))

        with pytest.raises(OntologyRoleError):
            select(ontology="hpo", explicit_path=wrong, cache_dir=cache)


class TestProvenanceIsUnchangedInShape:
    """Acceptance 30 — a selected input records what Phase 1 already records."""

    def test_the_build_records_the_file_it_was_given(self, tmp_path, select):
        from src.kg.provenance import source_entry
        from src.utils.fingerprint import file_sha256

        path = obo(tmp_path / "chosen.obo", version="releases/2026-09-01")
        loaded = select(explicit_path=path)

        entry = source_entry(
            role="mondo",
            path=loaded.source_path,
            digest=file_sha256(loaded.source_path),
            declared_version=loaded.declared_version,
        )

        assert entry["role"] == "mondo"
        assert entry["digest"] == file_sha256(path)
        assert entry["declared_version"] == "releases/2026-09-01"

    def test_the_digest_is_the_named_file_not_a_rival(self, tmp_path, select):
        """The point of selection reaching provenance: with two releases on
        disk, the record has to name the one that was used."""
        from src.utils.fingerprint import file_sha256

        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2020-01-01")
        wanted = obo(tmp_path / "wanted.obo", version="releases/2026-09-01")

        loaded = select(explicit_path=wanted, cache_dir=cache)

        assert file_sha256(loaded.source_path) == file_sha256(wanted)
        assert file_sha256(loaded.source_path) != file_sha256(cache / "mondo.obo")
