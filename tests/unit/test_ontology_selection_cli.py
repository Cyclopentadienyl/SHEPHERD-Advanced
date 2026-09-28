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

from src.ontology.loader import (  # noqa: E402
    OntologyFetchError,
    OntologyImportError,
    OntologyLoader,
)
from src.ontology.resolver import (  # noqa: E402
    AmbiguousOntologyError,
    OntologyResolutionError,
)
from src.ontology.roles import OntologyRoleError  # noqa: E402
from src.ontology.settings import OntologySettingsError  # noqa: E402
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
    # **Empty means nothing to fetch.** `{}` left the default PURLs in place,
    # so a regression in selection would have downloaded real MONDO during a
    # unit test and passed — on this sandbox the network is reachable.
    empty = tmp_path / "empty.yaml"
    empty.write_text(
        "ontology:\n  sources:\n    mondo: []\n    hpo: []\n    go: []\n    mp: []\n"
    )

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

        with pytest.raises(AmbiguousOntologyError) as caught:
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

        with pytest.raises(AmbiguousOntologyError, match="--mondo-path"):
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
            def _download_ontology(self, name, force, roots=()):
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

        with pytest.raises(OntologyResolutionError, match="two different instructions"):
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
            def _download_ontology(self, name, force, roots=()):
                calls.append((name, force))
                return fetched, fetched

        loaded = select(cache_dir=cache, force_download=True,
                        loader=Loader(cache_dir=cache))

        assert calls == [("mondo", True)], "the resolver short-circuited the flag"
        assert loaded.source_path == fetched

    def test_without_it_a_present_file_is_used_and_nothing_is_fetched(self, tmp_path, select):
        cache = tmp_path / "cache"
        present = obo(cache / "mondo.obo")

        class Loader(OntologyLoader):
            def _download_ontology(self, name, force, roots=()):
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

    def test_a_transfer_failure_is_reported_not_papered_over(self, tmp_path, select, monkeypatch):
        """**It has to reach the transfer.** The version of this test that stood
        here put a valid `mondo.obo` in the cache, so selection took it and the
        stubbed download was never called — it passed without exercising
        anything. The stale-cache fallback it was written for is gone: an
        acceptable file would have been selected without a download, and a file
        that was not selected was not selected for a reason, so there is
        nothing valid to fall back to. What a failed transfer owes the operator
        is a refusal that says so, names the roots, and points at the manual
        route."""
        import src.ontology.settings as settings_module
        from src.ontology import download as download_module
        from src.ontology.download import OntologyDownloadError
        from src.ontology.loader import OntologyFetchError

        config = tmp_path / "ok.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n      - https://purl.example/mondo.obo\n"
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(config))

        attempts = []

        def flaky(url, target, **kwargs):
            attempts.append(url)
            raise OntologyDownloadError("connection reset")

        monkeypatch.setattr(download_module, "download_ontology", flaky)
        cache = tmp_path / "cache"
        cache.mkdir()

        with pytest.raises(OntologyFetchError) as caught:
            select(cache_dir=cache, loader=OntologyLoader(cache_dir=cache))

        assert attempts == ["https://purl.example/mondo.obo"], "the transfer was never reached"
        message = str(caught.value)
        assert str(cache) in message, "the roots searched are not named"
        assert "manually" in message
        assert "refused by the destination policy" not in message, (
            "a transfer failure was reported as a configuration decision"
        )


def _config(tmp_path, body: str):
    """Point the settings reader at a YAML written for this test."""
    import src.ontology.settings as settings_module

    path = tmp_path / f"cfg{abs(hash(body))}.yaml"
    path.write_text(body)
    return path, settings_module


def _recording_download(body: bytes, calls: list):
    """A `download_ontology` stand-in that records where it was asked to write."""
    def fake(url, target, **kwargs):
        calls.append((url, Path(target)))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(body)
        return Path(target)
    return fake


HPO_BODY = (b"format-version: 1.2\ndata-version: hp/releases/2026-09-01\n"
            b"ontology: hp.obo\n\n[Term]\nid: HP:0000001\nname: p\n")
MONDO_NEW = (b"format-version: 1.2\ndata-version: releases/2026-09-01\n"
             b"ontology: mondo\n\n[Term]\nid: MONDO:0000001\nname: d\n")


class TestAMisfiledCacheFileIsNeverOverwritten:
    """The review's first P2, reproduced end to end before it was fixed.

    `<cache>/hpo.obo` declaring `mondo` was passed over for HPO and counted for
    MONDO; the HPO slot then found no candidate and downloaded over it. The
    MONDO slot's input was destroyed after it had been loaded, and provenance —
    hashed after both slots — recorded the replacement's digest against
    `mondo`. No role refusal fired, although §3.4 and acceptance 17 promise one.
    """

    @pytest.fixture
    def reachable_hpo(self, tmp_path, monkeypatch, select):
        from src.ontology import download as download_module

        path, settings_module = _config(
            tmp_path,
            "ontology:\n  sources:\n    hpo:\n      - https://mirror.test/hp.obo\n"
            "    mondo: []\n  allowed_hosts:\n    - mirror.test\n",
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(path))
        calls: list = []
        monkeypatch.setattr(download_module, "download_ontology",
                            _recording_download(HPO_BODY, calls))
        return calls

    def test_the_download_is_refused_and_the_file_left_alone(self, tmp_path, select, reachable_hpo):
        from src.ontology.loader import OntologyFetchError

        cache = tmp_path / "cache"
        misfiled = obo(cache / "hpo.obo", ontology="mondo", terms=("MONDO:0000001",))
        before = misfiled.read_bytes()

        with pytest.raises(OntologyFetchError) as caught:
            select(ontology="hpo", cache_dir=cache, loader=OntologyLoader(cache_dir=cache))

        assert misfiled.read_bytes() == before, "the misfiled file was overwritten"
        assert reachable_hpo == [], "something was fetched before the refusal"
        message = str(caught.value)
        assert "hpo.obo" in message and "mondo" in message, "the refusal does not say why"

    def test_in_build_order_the_mondo_input_survives_the_hpo_slot(self, tmp_path, select, reachable_hpo):
        """The sequence the build actually runs: MONDO loads the misfiled file,
        then HPO. Its digest has to still describe the bytes on disk."""
        from src.ontology.loader import OntologyFetchError
        from src.utils.fingerprint import file_sha256

        cache = tmp_path / "cache"
        misfiled = obo(cache / "hpo.obo", ontology="mondo", terms=("MONDO:0000001",))
        loader = OntologyLoader(cache_dir=cache)

        mondo = select(ontology="mondo", cache_dir=cache, loader=loader)
        loaded_digest = file_sha256(mondo.source_path)
        with pytest.raises(OntologyFetchError):
            select(ontology="hpo", cache_dir=cache, loader=loader)

        assert file_sha256(misfiled) == loaded_digest


class TestVerifyThenPublish:
    """A fresh copy replaces a working one only after it has passed."""

    @pytest.fixture
    def fresh(self, tmp_path, monkeypatch, select):
        from src.ontology import download as download_module

        path, settings_module = _config(
            tmp_path,
            "ontology:\n  sources:\n    mondo:\n      - https://mirror.test/mondo.obo\n"
            "  allowed_hosts:\n    - mirror.test\n",
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(path))
        calls: list = []

        def use(body):
            monkeypatch.setattr(download_module, "download_ontology",
                                _recording_download(body, calls))
            return calls
        return use

    def test_a_fresh_copy_that_fails_the_role_check_leaves_the_old_one(self, tmp_path, select, fresh):
        cache = tmp_path / "cache"
        old = obo(cache / "mondo.obo", version="releases/2025-01-01")
        before = old.read_bytes()
        fresh(HPO_BODY)

        with pytest.raises(OntologyRoleError):
            select(cache_dir=cache, force_download=True, loader=OntologyLoader(cache_dir=cache))

        assert old.read_bytes() == before, "a refused download replaced a working release"
        assert not [p.name for p in cache.iterdir() if p.name.endswith(".staged")]

    def test_a_fresh_copy_that_declares_an_import_leaves_the_old_one(self, tmp_path, select, fresh):
        from src.ontology.loader import OntologyImportError

        cache = tmp_path / "cache"
        old = obo(cache / "mondo.obo", version="releases/2025-01-01")
        before = old.read_bytes()
        fresh(MONDO_NEW.replace(b"ontology: mondo\n", b"ontology: mondo\nimport: http://x.invalid/y.obo\n"))

        with pytest.raises(OntologyImportError):
            select(cache_dir=cache, force_download=True, loader=OntologyLoader(cache_dir=cache))

        assert old.read_bytes() == before

    def test_a_good_fresh_copy_is_published_under_its_cache_name(self, tmp_path, select, fresh):
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2025-01-01")
        calls = fresh(MONDO_NEW)

        loaded = select(cache_dir=cache, force_download=True, loader=OntologyLoader(cache_dir=cache))

        assert calls and calls[0][1].name.startswith(".") and calls[0][1].name.endswith(".staged"), (
            "the download was not staged under a hidden name"
        )
        assert loaded.source_path == cache / "mondo.obo"
        assert (cache / "mondo.obo").read_bytes() == MONDO_NEW
        assert loaded.declared_version == "releases/2026-09-01"
        assert not [p.name for p in cache.iterdir() if p.name.endswith(".staged")]

    def test_a_fresh_copy_that_never_arrives_is_not_replaced_by_the_old_one(
        self, tmp_path, select, fresh, monkeypatch
    ):
        """`--force-download` against a source that fails. The cached file is
        exactly what was asked *not* to be used, so the build stops; serving it
        would report a fresh copy that never arrived."""
        from src.ontology import download as download_module
        from src.ontology.download import OntologyDownloadError
        from src.ontology.loader import OntologyFetchError

        cache = tmp_path / "cache"
        old = obo(cache / "mondo.obo", version="releases/2025-01-01")
        before = old.read_bytes()
        fresh(MONDO_NEW)  # the configuration; the transfer itself is replaced below

        def fails(url, target, **kwargs):
            raise OntologyDownloadError("connection reset")

        monkeypatch.setattr(download_module, "download_ontology", fails)

        with pytest.raises(OntologyFetchError, match="fresh copy was requested"):
            select(cache_dir=cache, force_download=True, loader=OntologyLoader(cache_dir=cache))

        assert old.read_bytes() == before

    def test_a_url_with_a_query_string_keeps_its_format(self, tmp_path, select, monkeypatch):
        """`mondo.owl?sig=abc` used to be written as `mondo.obo`, because the
        suffix was read from the whole URL rather than its path."""
        from src.ontology import download as download_module

        path, settings_module = _config(
            tmp_path,
            "ontology:\n  sources:\n    mondo:\n      - https://mirror.test/mondo.owl?sig=abc\n"
            "  allowed_hosts:\n    - mirror.test\n",
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(path))
        calls: list = []
        owl_body = owl(tmp_path / "src.owl").read_bytes()
        monkeypatch.setattr(download_module, "download_ontology",
                            _recording_download(owl_body, calls))
        cache = tmp_path / "cache"
        cache.mkdir()

        loaded = select(cache_dir=cache, loader=OntologyLoader(cache_dir=cache))

        assert calls[0][1].name.startswith(".mondo.owl.") and calls[0][1].name.endswith(".staged")
        assert loaded.source_path == cache / "mondo.owl"


class TestTwoFetchesAtOnce:
    """The reviewer's P2 on the staging name, with the interleaving pinned.

    Every fetch of one ontology into one cache used the same staging file. A
    verified MONDO, another fetch's HPO landed under that name, and the first
    renamed it into `mondo.obo` — publishing bytes nobody had checked, and
    recording (by hashing the path afterwards) a digest that did not describe
    what it had parsed. The synchronisation point is fixed: B runs to
    completion after A has verified and before A publishes."""

    @pytest.fixture
    def fetches(self, tmp_path, monkeypatch, select):
        from src.ontology import download as download_module

        path, settings_module = _config(
            tmp_path,
            "ontology:\n  sources:\n    mondo:\n      - https://mirror.test/mondo.obo\n"
            "  allowed_hosts:\n    - mirror.test\n",
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(path))
        bodies: list = []
        staged: list = []

        def deliver(url, target, **kwargs):
            staged.append(Path(target))
            Path(target).write_bytes(bodies.pop(0))
            return Path(target)

        monkeypatch.setattr(download_module, "download_ontology", deliver)
        cache = tmp_path / "cache"
        cache.mkdir()

        def between_verify_and_publish(run_b):
            """Run `run_b` right after A's staged file has passed `load`."""
            genuine = OntologyLoader.load
            state = {"fired": False}

            def load(self, path, expect=None):
                result = genuine(self, path, expect=expect)
                if not state["fired"] and Path(path).name.endswith(".staged"):
                    state["fired"] = True
                    run_b()
                return result

            monkeypatch.setattr(OntologyLoader, "load", load)

        return cache, bodies, staged, between_verify_and_publish

    def test_a_rejected_fetch_cannot_become_the_published_file(self, fetches):
        import hashlib

        cache, bodies, staged, between = fetches
        bodies[:] = [MONDO_NEW, HPO_BODY]  # A gets MONDO, B gets HPO
        outcome = {}

        def b_fetches_hpo_bytes_into_the_mondo_slot():
            with pytest.raises(OntologyRoleError):
                OntologyLoader(cache_dir=cache)._fetch_ontology("mondo", True, roots=[cache])
            outcome["b_refused"] = True

        between(b_fetches_hpo_bytes_into_the_mondo_slot)
        a = OntologyLoader(cache_dir=cache)._fetch_ontology("mondo", True, roots=[cache])

        assert outcome == {"b_refused": True}
        assert len({p.name for p in staged}) == 2, "the two fetches shared a staging file"
        assert (cache / "mondo.obo").read_bytes() == MONDO_NEW, "unverified bytes were published"
        assert a.declared_version == "releases/2026-09-01"
        assert a.source_digest == hashlib.sha256(MONDO_NEW).hexdigest()
        assert sorted(p.name for p in cache.iterdir()) == ["mondo.obo"], (
            "a staging file was left behind, or B removed A's"
        )

    def test_each_fetch_carries_the_digest_of_what_it_verified(self, fetches):
        """Both succeed, with different releases, and B publishes in between.
        The cache ends up holding whichever was published last — both were
        verified — and each build's ontology names the bytes *it* parsed,
        not whatever the cache name held when someone hashed it."""
        import hashlib

        cache, bodies, staged, between = fetches
        mondo_b = MONDO_NEW.replace(b"releases/2026-09-01", b"releases/2026-10-01")
        bodies[:] = [MONDO_NEW, mondo_b]
        seen = {}

        def b_publishes_another_release():
            seen["b"] = OntologyLoader(cache_dir=cache)._fetch_ontology("mondo", True, roots=[cache])

        between(b_publishes_another_release)
        a = OntologyLoader(cache_dir=cache)._fetch_ontology("mondo", True, roots=[cache])

        assert a.source_digest == hashlib.sha256(MONDO_NEW).hexdigest()
        assert seen["b"].source_digest == hashlib.sha256(mondo_b).hexdigest()
        assert (cache / "mondo.obo").read_bytes() == MONDO_NEW  # A published last
        assert sorted(p.name for p in cache.iterdir()) == ["mondo.obo"]


class TestNothingToFetchIsSaidPlainly:

    def test_no_source_names_the_roots_and_says_nothing_was_tried(self, tmp_path, select):
        """It used to say a download failed and that URLs "may be outdated" when
        nothing had been attempted, and named no root, although §3.1 and both
        guides promise the roots searched."""
        from src.ontology.loader import OntologyFetchError

        cache = tmp_path / "cache"
        cache.mkdir()

        with pytest.raises(OntologyFetchError) as caught:
            select(cache_dir=cache, loader=OntologyLoader(cache_dir=cache))

        message = str(caught.value)
        assert str(cache) in message
        assert "nothing was attempted" in message
        assert "outdated" not in message


class TestTheNamedLoadersSearchTheConfiguredRoots:

    def test_load_mondo_finds_a_file_under_ontology_roots(self, tmp_path, monkeypatch):
        """They looked only in the cache, so a library caller on a deployment
        with `paths.ontology_roots` fetched over the network while the
        configured file sat there — and the fetched copy then made the next
        build refuse as ambiguous."""
        root = tmp_path / "configured"
        staged = obo(root / "mondo.obo")
        path, settings_module = _config(
            tmp_path,
            f"paths:\n  ontology_roots:\n    - {root}\n"
            "ontology:\n  sources:\n    mondo: []\n",
        )
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: _genuine_settings(path))

        class Loader(OntologyLoader):
            def _download_ontology(self, name, force, roots=()):
                raise AssertionError("fetched while a configured file was present")

        assert Loader(cache_dir=tmp_path / "cache").load_mondo().source_path == staged


class TestTheBuildEntryPoint:

    def _build_args(self, tmp_path):
        external = tmp_path / "external"
        external.mkdir()
        for name in ("phenotype.hpoa", "genes_to_phenotype.txt"):
            (external / name).write_text("")
        return external

    def test_a_path_with_force_download_is_refused_before_anything_is_fetched(
        self, tmp_path, monkeypatch, select
    ):
        """It was checked per slot, so `--hpo-path` with `--force-download`
        refused only after MONDO had been fetched and its cache replaced."""
        module = build_module()

        class Detonator:
            def __init__(self, *a, **k):
                raise AssertionError("a loader was constructed, so a fetch could follow")

        monkeypatch.setattr(module, "OntologyLoader", Detonator)
        hpo = obo(tmp_path / "hp.obo", ontology="hpo", terms=("HP:1",))

        with pytest.raises(OntologyResolutionError, match="Nothing has been fetched"):
            module.build_knowledge_graph(
                external_dir=self._build_args(tmp_path), workspace=tmp_path / "ws",
                hpo_path=hpo, force_download=True,
            )

    def test_a_refusal_is_an_exception_a_caller_can_catch(self, tmp_path, select):
        """`probe_deployment` calls the build as a function and catches
        `Exception`. `SystemExit` is not one, so the old refusal ended the
        process and lost the probe's whole report."""
        cache = tmp_path / "cache"
        obo(cache / "mondo.obo", version="releases/2026-09-01")
        owl(cache / "mondo.owl")

        with pytest.raises(Exception) as caught:
            select(cache_dir=cache)

        assert isinstance(caught.value, AmbiguousOntologyError)
        assert not isinstance(caught.value, SystemExit)

    @pytest.mark.parametrize("refusal", [
        AmbiguousOntologyError("two candidates", ()),
        OntologyRoleError("two candidates"),
        OntologyImportError("two candidates"),
        OntologyFetchError("two candidates"),
        OntologySettingsError("two candidates"),
    ], ids=["ambiguous", "role", "imports", "fetch", "settings"])
    def test_main_turns_a_refusal_into_an_exit_status(self, monkeypatch, capsys, refusal):
        """Every ontology refusal, a malformed configuration included — each is
        written for an operator, and a traceback buries the sentence."""
        module = build_module()

        def refuse(args):
            raise refusal

        monkeypatch.setattr(module, "_run_build", refuse)
        monkeypatch.setattr(sys, "argv", ["build_knowledge_graph.py",
                                          "--external-dir", "x", "--workspace", "y"])

        with pytest.raises(SystemExit) as caught:
            module.main()

        assert caught.value.code == 2
        assert "two candidates" in capsys.readouterr().err

    def test_the_probe_passes_the_paths_through(self, tmp_path, monkeypatch):
        """Both phase-F builds, observed rather than read from the source: a
        probe that dropped the paths would build from whatever the roots held,
        which is the ambiguity this phase refuses."""
        import scripts.build_knowledge_graph as build

        spec = importlib.util.spec_from_file_location(
            "probe_deployment", REPO / "scripts" / "probe_deployment.py")
        probe = importlib.util.module_from_spec(spec)
        # Its dataclasses resolve their own module by name while being built.
        sys.modules[spec.name] = probe
        try:
            spec.loader.exec_module(probe)
        finally:
            sys.modules.pop(spec.name, None)

        args = probe.parse_args(["--work-dir", "w", "--mondo-path", "m.obo", "--hpo-path", "h.obo"])
        assert args.mondo_path == Path("m.obo") and args.hpo_path == Path("h.obo")

        calls = []

        def record(**kwargs):
            calls.append((kwargs.get("mondo_path"), kwargs.get("hpo_path")))
            # Enough for the second probe to run, and nothing more.
            workspace = Path(kwargs["workspace"])
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "kg.json").write_text("{}")
            raise RuntimeError("recorded, not built")

        monkeypatch.setattr(build, "build_knowledge_graph", record)
        external = tmp_path / "external"
        external.mkdir()
        for name in ("phenotype.hpoa", "genes_to_phenotype.txt"):
            (external / name).write_text("")

        probe.phase_real_build(probe.Report(), tmp_path / "work", external, 1, 1,
                               mondo_path=Path("m.obo"), hpo_path=Path("h.obo"))

        assert calls == [(Path("m.obo"), Path("h.obo"))] * 2
