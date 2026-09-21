"""Where this server may fetch an ontology from, and where the rule is enforced.

`PLAN_ONTOLOGY_PHASE2.md` §3.5 and §3.5.1; acceptance 21-29.

**The rule has to live at the request, not at the setting**, and the reason is
specific to this project rather than general caution: every source configured
here is an OBO Foundry PURL, and a PURL *is* a redirect. A check performed when
the list is read has already passed by the time the 302 arrives. Measured on
this interpreter, `HTTPRedirectHandler.http_error_302` permits targets whose
scheme is in ``('http', 'https', 'ftp', '')`` — so `https` → `ftp` is followed.

**No test here touches DNS or any host.** A stub resolver supplies the
addresses, which is what makes "a host resolving to several addresses, one of
them private" a case that can be written at all.

Module: tests/unit/test_ontology_download_policy.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ontology.download import (  # noqa: E402
    ALLOWED_SCHEMES,
    DestinationPolicy,
    OntologyDownloadError,
    check_destination,
    download_ontology,
)
from src.ontology.settings import (  # noqa: E402
    DEFAULT_ONTOLOGY_SOURCES,
    OntologySettingsError,
    load_ontology_settings,
)

PUBLIC = "93.184.216.34"
PRIVATE = "10.1.2.3"
LOOPBACK = "127.0.0.1"
METADATA = "169.254.169.254"


def policy(mapping=None, *, allowed=()):
    """A policy whose DNS is a dictionary."""
    mapping = mapping or {}
    return DestinationPolicy(
        allowed_hosts=tuple(allowed),
        resolver=lambda host: mapping.get(host, [PUBLIC]),
    )


class TestTheSchemeRule:

    def test_a_permitted_scheme_passes(self):
        check_destination("https://example.org/mondo.obo", policy())

    @pytest.mark.parametrize("url", [
        "ftp://example.org/mondo.obo",
        "file:///etc/passwd",
        "data:text/plain,hello",
        "example.org/mondo.obo",
    ])
    def test_anything_else_is_refused(self, url):
        """`urlretrieve`'s opener carries `FileHandler`, `FTPHandler` and
        `DataHandler`, so a field named "download URL" accepts more than a URL
        unless it says otherwise."""
        with pytest.raises(OntologyDownloadError, match="scheme"):
            check_destination(url, policy())

    def test_a_url_with_no_host_is_refused(self):
        with pytest.raises(OntologyDownloadError, match="no host"):
            check_destination("https:///mondo.obo", policy())

    def test_the_settings_reader_asks_the_same_module(self, tmp_path):
        """**Acceptance 21**, and one rule rather than two: the config-time
        check imports `ALLOWED_SCHEMES` instead of restating it."""
        config = tmp_path / "deployment.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n      - ftp://example.org/mondo.obo\n"
        )

        with pytest.raises(OntologySettingsError) as caught:
            load_ontology_settings(config)

        assert "ftp://example.org/mondo.obo" in str(caught.value)
        assert "ontology.sources.mondo" in str(caught.value)


class TestTheDestinationRule:

    def test_a_public_address_is_permitted(self):
        """**Acceptance 27's mirror**: without this the default would be a
        blanket block and every refusal below would pass vacuously."""
        check_destination("https://purl.obolibrary.org/obo/mondo.obo",
                          policy({"purl.obolibrary.org": [PUBLIC]}))

    @pytest.mark.parametrize("address,what", [
        (PRIVATE, "private"),
        (LOOPBACK, "loopback"),
        (METADATA, "link-local"),
        ("::1", "IPv6 loopback"),
        ("fd00::1", "IPv6 private"),
        ("0.0.0.0", "unspecified"),
    ])
    def test_an_address_inside_the_network_is_refused(self, address, what):
        """**Acceptance 25.** `is_global` is one predicate over both families."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination("https://mirror.internal/mondo.obo",
                              policy({"mirror.internal": [address]}))

    def test_a_literal_private_address_is_refused(self):
        """**Acceptance 26.** It has no hostname to match against a list, which
        is why the rule is on addresses rather than on spelling."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination(f"http://{METADATA}/latest/meta-data/", policy())

    def test_an_allowed_host_is_permitted_inside_the_network(self):
        """**Acceptance 27.** An in-house mirror is legitimate and probably
        desirable; the entry is what makes it deliberate."""
        check_destination(
            "https://ontology-mirror.hospital.internal/mondo.obo",
            policy({"ontology-mirror.hospital.internal": [PRIVATE]},
                   allowed=("ontology-mirror.hospital.internal",)),
        )

    def test_the_allow_list_is_not_case_sensitive(self):
        check_destination(
            "https://Mirror.Internal/mondo.obo",
            policy({"Mirror.Internal": [PRIVATE]}, allowed=("mirror.internal",)),
        )

    def test_one_private_answer_among_several_refuses(self):
        """**Acceptance 29, and the whole gap in one line.** A name resolving to
        a public address *and* a private one is a name that can reach the
        private one; admitting it because the first answer was fine is exactly
        the mistake."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination("https://split.example/mondo.obo",
                              policy({"split.example": [PUBLIC, PRIVATE]}))

    def test_a_host_that_resolves_to_nothing_is_refused(self):
        """Unknown is not permitted: where the request would go is unknown,
        which is not the same as its being acceptable."""
        with pytest.raises(OntologyDownloadError, match="no address"):
            check_destination("https://nowhere.example/x.obo",
                              policy({"nowhere.example": []}))

    def test_an_unclassifiable_answer_is_refused(self):
        with pytest.raises(OntologyDownloadError, match="classify"):
            check_destination("https://odd.example/x.obo",
                              policy({"odd.example": ["not-an-address"]}))


class TestTheRedirectIsWhereTheRuleActuallyBites:
    """Acceptance 22, 23 and 28. The stdlib handler is the measured hazard."""

    @staticmethod
    def _handler(pol):
        from src.ontology.download import _GuardedRedirectHandler

        return _GuardedRedirectHandler(pol)

    class _Resp:
        headers: dict = {}

        def geturl(self):
            return "https://purl.example/obo/mondo.obo"

    def test_the_stdlib_would_follow_https_to_ftp(self):
        """The premise, measured here so the guard below is not defending
        against something that never happens."""
        import inspect
        import re

        source = inspect.getsource(urllib.request.HTTPRedirectHandler.http_error_302)
        permitted = re.search(r"urlparts\.scheme not in \(([^)]*)\)", source)

        assert permitted is not None
        assert "'ftp'" in permitted.group(1), (
            "this interpreter no longer permits ftp redirect targets; the guard "
            "is still correct but this test's premise has changed"
        )

    def test_a_redirect_to_ftp_is_refused(self):
        """**Acceptance 22.** The config-time check passed long ago."""
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")

        with pytest.raises(OntologyDownloadError, match="scheme"):
            self._handler(policy()).redirect_request(
                req, self._Resp(), 302, "Found", {}, "ftp://elsewhere.example/mondo.obo"
            )

    def test_a_redirect_into_the_network_is_refused(self):
        """**Acceptance 28**, the destination twin of the case above."""
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")
        pol = policy({"purl.example": [PUBLIC], "inside.example": [PRIVATE]})

        with pytest.raises(OntologyDownloadError, match="globally routable"):
            self._handler(pol).redirect_request(
                req, self._Resp(), 302, "Found", {}, "https://inside.example/mondo.obo"
            )

    def test_an_ordinary_redirect_is_still_followed(self):
        """**Acceptance 23.** Every source this project ships is a PURL, so a
        guard that blocked redirects would block everything."""
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")
        pol = policy({"purl.example": [PUBLIC], "github.example": [PUBLIC]})

        result = self._handler(pol).redirect_request(
            req, self._Resp(), 302, "Found", {}, "https://github.example/mondo.obo"
        )

        assert result is not None
        assert result.full_url == "https://github.example/mondo.obo"

    def test_the_chain_is_bounded(self):
        from src.ontology.download import MAX_REDIRECTS

        handler = self._handler(policy())
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")

        with pytest.raises(OntologyDownloadError, match="redirects"):
            for hop in range(MAX_REDIRECTS + 2):
                handler.redirect_request(
                    req, self._Resp(), 302, "Found", {},
                    f"https://hop{hop}.example/mondo.obo",
                )


class TestTheOneFetchPath:

    def test_a_refused_source_writes_nothing(self, tmp_path):
        target = tmp_path / "mondo.obo"

        with pytest.raises(OntologyDownloadError):
            download_ontology("ftp://example.org/mondo.obo", target, policy=policy())

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_refusal_leaves_an_existing_file_untouched(self, tmp_path):
        """A refused fetch is not a reason to truncate what is already serving."""
        target = tmp_path / "mondo.obo"
        target.write_bytes(b"the file that is already here")

        with pytest.raises(OntologyDownloadError):
            download_ontology("https://blocked.example/mondo.obo", target,
                              policy=policy({"blocked.example": [PRIVATE]}))

        assert target.read_bytes() == b"the file that is already here"

    def test_a_permitted_fetch_writes_the_bytes(self, tmp_path):
        """With a stub opener, so nothing leaves this process."""
        target = tmp_path / "mondo.obo"
        payload = b"format-version: 1.2\n"

        class Stub:
            def open(self, url, timeout=None):
                class Body:
                    def __init__(self):
                        self._data = payload

                    def read(self, n):
                        out, self._data = self._data[:n], self._data[n:]
                        return out

                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        return False

                return Body()

        written = download_ontology(
            "https://purl.example/obo/mondo.obo", target,
            policy=policy({"purl.example": [PUBLIC]}),
            opener_factory=lambda *handlers: Stub(),
        )

        assert written.read_bytes() == payload

    def test_a_failed_transfer_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "mondo.obo"

        class Stub:
            def open(self, url, timeout=None):
                raise OSError("connection reset")

        with pytest.raises(OntologyDownloadError, match="failed"):
            download_ontology("https://purl.example/obo/mondo.obo", target,
                              policy=policy({"purl.example": [PUBLIC]}),
                              opener_factory=lambda *h: Stub())

        assert not target.exists()
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]

    def test_the_owl_fallback_has_no_fetch_of_its_own(self):
        """**Acceptance 24.** A fallback with its own fetch is the gate not
        existing — and the fallback is the path a rotted PURL leads to, so it
        is the one that most needs the rules."""
        import inspect

        from src.ontology import loader

        source = inspect.getsource(loader)
        assert "urlretrieve" not in source, (
            "the loader fetches directly again, bypassing the scheme and "
            "destination rules"
        )
        body = inspect.getsource(loader.OntologyLoader._download_ontology)
        calls = [line for line in body.splitlines() if "download_ontology(url" in line]
        assert len(calls) == 1, f"more than one fetch call in the download path: {calls}"


class TestTheSettingsContract:

    def test_an_absent_file_is_absence_not_an_error(self, tmp_path):
        settings = load_ontology_settings(tmp_path / "nothing.yaml")

        assert settings.roots == ()
        assert settings.urls_for("mondo") == DEFAULT_ONTOLOGY_SOURCES["mondo"]

    def test_relative_roots_resolve_against_the_repository(self, tmp_path):
        """A build run from a subdirectory has to find the same files as one
        run from the root, so the base is stated rather than inherited from
        the working directory."""
        from src.ontology.settings import REPO_ROOT

        config = tmp_path / "deployment.yaml"
        config.write_text("paths:\n  ontology_roots:\n    - data/ontologies/\n")

        settings = load_ontology_settings(config)

        assert settings.roots == (REPO_ROOT / "data/ontologies",)

    def test_an_absolute_root_is_left_alone(self, tmp_path):
        config = tmp_path / "deployment.yaml"
        config.write_text(f"paths:\n  ontology_roots:\n    - {tmp_path / 'onts'}\n")

        assert load_ontology_settings(config).roots == (tmp_path / "onts",)

    def test_a_present_but_malformed_key_raises(self, tmp_path):
        """**Absent and wrong are different states** and only one of them is a
        deployment as intended."""
        config = tmp_path / "deployment.yaml"
        config.write_text("paths:\n  ontology_roots: data/ontologies/\n")

        with pytest.raises(OntologySettingsError, match="must be a list"):
            load_ontology_settings(config)

    def test_a_configured_source_replaces_the_default(self, tmp_path):
        """**Acceptance 6's config half**: editing a URL here changes what a
        download attempts, with no code change."""
        config = tmp_path / "deployment.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n"
            "      - https://mirror.example/mondo.obo\n"
        )

        settings = load_ontology_settings(config)

        assert settings.urls_for("mondo") == ("https://mirror.example/mondo.obo",)
        assert settings.urls_for("hpo") == DEFAULT_ONTOLOGY_SOURCES["hpo"], (
            "listing one ontology should not silently drop the others"
        )

    def test_allowed_hosts_are_read(self, tmp_path):
        config = tmp_path / "deployment.yaml"
        config.write_text("ontology:\n  allowed_hosts:\n    - mirror.internal\n")

        assert load_ontology_settings(config).allowed_hosts == ("mirror.internal",)

    def test_the_committed_configuration_still_parses(self):
        """The file this project actually ships, read through this reader."""
        settings = load_ontology_settings()

        assert settings.urls_for("mondo")
        for name in DEFAULT_ONTOLOGY_SOURCES:
            for url in settings.urls_for(name):
                assert url.split(":", 1)[0] in ALLOWED_SCHEMES


class TestATruncatedTransferIsNotAFinishedOne:
    """The check `urlretrieve` had and this downloader dropped.

    `HTTPResponse.read(amt)` returns `b""` when the connection closes early —
    it does not raise — so a short read ends the loop exactly like a complete
    one, and temp-and-replace then publishes the fragment over a good file. The
    cut can land on a stanza boundary, in which case the fragment parses, passes
    the role check, and has its digest recorded as the input.
    """

    @staticmethod
    def _opener(body: bytes, declared: int | None):
        """A real `http.client.HTTPResponse` over a controlled socket.

        Not a stub that raises: the whole point is that nothing raises. A test
        whose opener threw `OSError` would pass against the broken version.
        """
        import http.client
        import io

        header = b"HTTP/1.1 200 OK\r\n"
        header += (
            f"Content-Length: {declared}\r\n".encode()
            if declared is not None
            else b"Transfer-Encoding: chunked\r\n"
        )
        raw = header + b"\r\n" + body

        class Sock:
            def __init__(self):
                self._f = io.BytesIO(raw)

            def makefile(self, *a, **k):
                return self._f

        class Opener:
            def open(self, url, timeout=None):
                response = http.client.HTTPResponse(Sock(), method="GET")
                response.begin()
                return response

        return lambda *handlers: Opener()

    def test_a_short_body_is_refused(self, tmp_path):
        target = tmp_path / "mondo.obo"

        with pytest.raises(OntologyDownloadError, match="delivered"):
            download_ontology(
                "https://purl.example/obo/mondo.obo", target,
                policy=policy({"purl.example": [PUBLIC]}),
                opener_factory=self._opener(b"format-version: 1.2\n", 120),
            )

        assert not target.exists()

    def test_the_file_already_in_place_survives_it(self, tmp_path):
        """**The damage the missing check actually did.** A stanza-aligned cut
        parses, so the fragment would have replaced a complete ontology and
        been recorded as the input."""
        target = tmp_path / "mondo.obo"
        good = b"format-version: 1.2\nontology: mondo\n\n[Term]\nid: MONDO:1\nname: t\n"
        target.write_bytes(good)

        with pytest.raises(OntologyDownloadError):
            download_ontology(
                "https://purl.example/obo/mondo.obo", target,
                policy=policy({"purl.example": [PUBLIC]}),
                opener_factory=self._opener(b"format-version: 1.2\n[Term]\n", 900),
            )

        assert target.read_bytes() == good
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]

    def test_a_complete_body_is_accepted(self, tmp_path):
        """Or the check above would be satisfied by refusing everything."""
        target = tmp_path / "mondo.obo"
        body = b"format-version: 1.2\n"

        download_ontology(
            "https://purl.example/obo/mondo.obo", target,
            policy=policy({"purl.example": [PUBLIC]}),
            opener_factory=self._opener(body, len(body)),
        )

        assert target.read_bytes() == body

    def test_a_response_declaring_no_length_is_not_failed_for_it(self, tmp_path):
        """**Not possible to check is not the same as checked.** A chunked
        response carries no `Content-Length`, and treating the absence as zero
        would make every such transfer pass a check that never ran — while
        treating it as failure would refuse a legitimate server."""
        target = tmp_path / "mondo.obo"

        download_ontology(
            "https://purl.example/obo/mondo.obo", target,
            policy=policy({"purl.example": [PUBLIC]}),
            opener_factory=self._opener(b"0\r\n\r\n", None),
        )

        assert target.exists()
