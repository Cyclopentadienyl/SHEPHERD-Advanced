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


class TestTheNamedLoadersUseTheSameSelection:
    """`load_mondo()` / `load_hpo()` are still public, and were a second rule.

    They opened `<cache_dir>/<name>.obo` and fell back to `<name>.owl`, so a
    cache holding both took the OBO **silently** — the implicit precedence
    §3.1.2 declines to reinstate, still live on the library path while the CLI
    refused. And they never passed the role they obviously knew, so the §3.4
    check was reachable only from the build script.

    Two semantics for one question is the parallel pipeline this phase exists to
    avoid, so these enter through the public API rather than through
    `load(..., expect=...)`.
    """

    @pytest.fixture
    def offline(self, tmp_path, monkeypatch):
        """No configured sources, so a miss cannot become a real download."""
        import src.ontology.settings as settings_module

        config = tmp_path / "offline.yaml"
        config.write_text("ontology:\n  sources:\n    mondo: []\n    hpo: []\n")
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(config))

    def test_load_hpo_applies_the_role_check(self, tmp_path, offline):
        cache = tmp_path / "cache"
        obo(cache / "hpo.obo", ontology="mondo", terms=("MONDO:0000001",))

        with pytest.raises((OntologyRoleError, RuntimeError)):
            OntologyLoader(cache_dir=cache).load_hpo()

    def test_load_hpo_still_loads_a_correct_file(self, tmp_path, offline):
        """Or the refusal above is a loader that rejects everything."""
        cache = tmp_path / "cache"
        obo(cache / "hpo.obo", ontology="hpo", terms=("HP:0000001",))

        assert OntologyLoader(cache_dir=cache).load_hpo().num_terms == 1

    def test_load_mondo_refuses_an_ambiguous_cache_like_the_cli(self, tmp_path, offline):
        from src.ontology.resolver import AmbiguousOntologyError

        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2026-09-01")
        owl(cache / "mondo.owl")

        with pytest.raises(AmbiguousOntologyError):
            OntologyLoader(cache_dir=cache).load_mondo()

    def test_the_owl_fixture_is_a_file_the_real_loader_accepts(self, tmp_path, offline):
        """**The fixture has to be real or it proves nothing.** An earlier one
        emitted undeclared XML prefixes that no reader would open, which is how
        the scanner's namespace blindness survived a passing test."""
        path = owl(tmp_path / "mondo.owl", classes=2)

        loaded = OntologyLoader(cache_dir=tmp_path / "cache").load(path, expect="mondo")

        assert loaded.num_terms >= 1


class TestAPolicyRefusalIsNotAStaleCacheSuccess:
    """§3.5's refusals reaching the operator instead of being logged past.

    Every `OntologyDownloadError` used to be collected as a warning and the old
    cache returned — so a destination this server may **never** fetch from was
    reported as a successful build, on an input nobody asked for. The function's
    own docstring said a refused destination is not a reason to fall back while
    the code did exactly that.
    """

    @pytest.fixture
    def blocked(self, tmp_path, monkeypatch, select):
        """**Depends on `select` so it patches last.** `select` holds the
        configuration empty; a fixture that ran before it would be overwritten,
        the default PURLs would be used, and on a machine with a network the
        test would quietly fetch the real ontology and prove nothing."""
        import src.ontology.settings as settings_module

        config = tmp_path / "blocked.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n      - http://127.0.0.1/blocked.obo\n"
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(config))

    def test_force_download_against_a_refused_source_does_not_serve_the_cache(
        self, tmp_path, blocked, select
    ):
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2019-01-01")

        with pytest.raises(RuntimeError, match="refused by the destination policy"):
            select(cache_dir=cache, force_download=True,
                   loader=OntologyLoader(cache_dir=cache))

    def test_the_refusal_names_the_remedy(self, tmp_path, blocked, select):
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo")

        with pytest.raises(RuntimeError) as caught:
            select(cache_dir=cache, force_download=True,
                   loader=OntologyLoader(cache_dir=cache))

        message = str(caught.value)
        assert "allowed_hosts" in message
        assert "127.0.0.1" in message

    def test_a_transfer_failure_may_still_fall_back(self, tmp_path, select, monkeypatch):
        """**The distinction, not a blanket refusal.** A transfer that failed
        may succeed next time; a destination the policy forbids will not. Only
        the second is a configuration error to report."""
        import src.ontology.settings as settings_module
        from src.ontology import download as download_module
        from src.ontology.download import OntologyDownloadError

        config = tmp_path / "ok.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n      - https://purl.example/mondo.obo\n"
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(config))

        def flaky(url, target, **kwargs):
            raise OntologyDownloadError("connection reset")

        monkeypatch.setattr(download_module, "download_ontology", flaky)

        cache = tmp_path / "cache"
        present = obo(cache / "mondo.obo", version="releases/2019-01-01")

        loaded = select(cache_dir=cache, loader=OntologyLoader(cache_dir=cache))

        assert loaded.source_path == present
