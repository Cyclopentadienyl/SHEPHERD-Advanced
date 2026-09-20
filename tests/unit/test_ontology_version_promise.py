"""The `version` argument stops meaning something it never did.

`_load_known_ontology` folded `version` into an in-memory cache key and then
opened `<cache_dir>/<name>.obo` regardless, so `load_mondo(version="2026-01-01")`
returned whatever that file happened to be — a different release, silently,
under the name of the one asked for. A knowledge graph then carries that choice
into every index it assigns, which is why a quiet substitution here is not a
cosmetic problem.

Selecting a release is later work. Until it exists the honest behaviour is to
refuse.

**`force_download` is a different control and is deliberately untouched.** An
earlier draft of the plan listed the two together as inert; review established
that `force_download` is checked at all three branches and works. The last class
here exists so that a future tidy-up of "the unused ontology flags" cannot
remove it without a test failing.

Module: tests/unit/test_ontology_version_promise.py
"""
from __future__ import annotations

import pytest


@pytest.fixture
def loader(tmp_path, monkeypatch):
    """A loader whose cache holds a file, with parsing and download replaced.

    Nothing here reaches the network or pronto: the point is which branch runs,
    not what an ontology contains.
    """
    from src.ontology.loader import OntologyLoader

    (tmp_path / "mondo.obo").write_text("format-version: 1.2\n")
    (tmp_path / "hpo.obo").write_text("format-version: 1.2\n")

    instance = OntologyLoader(cache_dir=tmp_path)
    calls = {"download": 0, "load": 0}

    def _fake_download(name, force):
        calls["download"] += 1
        return tmp_path / f"{name}.obo"

    class _Parsed:
        """Accepts the attributes the loader sets after parsing; a bare
        `object()` cannot, which is the only thing this needs to be."""

    def _fake_load(path):
        calls["load"] += 1
        return _Parsed()

    monkeypatch.setattr(instance, "_download_ontology", _fake_download)
    monkeypatch.setattr(instance, "load", _fake_load)
    return instance, calls, tmp_path


class TestAVersionItCannotHonourIsRefused:

    @pytest.mark.parametrize("method", ["load_mondo", "load_hpo", "load_go", "load_mp"])
    def test_every_loader_refuses_a_specific_release(self, loader, method):
        """All four, because the rule lives in the shared path and a future
        ontology gets it for free only if that stays true."""
        instance, _, _ = loader

        with pytest.raises(ValueError, match="can only honour"):
            getattr(instance, method)(version="2026-01-01")

    def test_the_refusal_names_what_was_asked_for(self, loader):
        instance, _, _ = loader

        with pytest.raises(ValueError, match="2026-01-01"):
            instance.load_mondo(version="2026-01-01")

    def test_nothing_is_read_or_fetched_before_the_refusal(self, loader):
        """A pre-check, not a post-hoc complaint: refusing after a 50 MB
        download would be a worse version of the same bug."""
        instance, calls, _ = loader

        with pytest.raises(ValueError):
            instance.load_mondo(version="v5")

        assert calls == {"download": 0, "load": 0}

    def test_two_different_versions_no_longer_pass_for_two_things(self, loader):
        """What the argument actually did. Both strings resolved to one file,
        so a caller comparing results across versions compared a file with
        itself."""
        instance, _, _ = loader

        for version in ("2026-01-01", "2025-01-01"):
            with pytest.raises(ValueError):
                instance.load_mondo(version=version)


class TestTheDefaultStillWorks:
    """Without this, the refusals above hold for a loader that refuses
    everything — and every real caller passes no version at all."""

    def test_the_cached_file_is_used_with_no_download(self, loader):
        instance, calls, _ = loader

        instance.load_mondo()

        assert calls["load"] == 1
        assert calls["download"] == 0

    def test_latest_is_accepted_explicitly_too(self, loader):
        instance, calls, _ = loader

        instance.load_hpo(version="latest")

        assert calls["load"] == 1


class TestForceDownloadIsNotCollateralDamage:
    """The control this change nearly removed.

    The plan's first draft called `force_download` inert because no caller
    passes it. It is checked at the memory cache, the disk cache and the
    download decision, and passing True re-fetches. "Nothing calls it" and "it
    does nothing" are different claims.

    **Two of its three guards are mutually redundant, and these tests say so
    rather than pretending otherwise.** Removing the disk-cache guard alone
    still downloads, because the download branch's `or force_download` fires;
    removing the download branch's alone still downloads, because the disk guard
    left `cache_file` as None. Each single removal is an equivalent mutation and
    no test can kill it. Removing **both** is not equivalent, and it is killed
    here — which is the case that matters, because it is what deleting
    `force_download` support looks like. The memory-cache guard is not redundant
    and is killed on its own.
    """

    def test_it_still_bypasses_a_present_cache_file(self, loader):
        instance, calls, _ = loader

        instance.load_mondo(force_download=True)

        assert calls["download"] == 1, (
            "force_download no longer reaches the download branch"
        )

    def test_it_still_bypasses_the_memory_cache(self, loader):
        instance, calls, _ = loader

        instance.load_mondo()
        assert calls["download"] == 0
        instance.load_mondo(force_download=True)

        assert calls["download"] == 1, (
            "the second call was served from memory despite force_download"
        )

    def test_without_it_the_memory_cache_is_used(self, loader):
        """The other half: the cache must still be a cache."""
        instance, calls, _ = loader

        instance.load_mondo()
        instance.load_mondo()

        assert calls["load"] == 1, "the second call re-parsed instead of caching"
